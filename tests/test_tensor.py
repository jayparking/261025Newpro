"""Meaningful CPU tensor tests; skip explicitly when optional packages are absent."""
import copy
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
HAS_HF = HAS_TORCH and all(importlib.util.find_spec(n) for n in ("transformers", "peft", "tokenizers"))
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "PyTorch/NumPy unavailable; install requirements on server")
class TensorTests(unittest.TestCase):
    def test_microbatch_gradient_scaling_variable_lengths(self):
        import torch
        from grpo_order.objective import accumulate_group
        from grpo_order.core import advantages, load_config
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(12, 8)
                self.head = torch.nn.Linear(8, 12)
            def forward(self, input_ids, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.head(self.embed(input_ids)))
        torch.manual_seed(9)
        base = Tiny()
        responses = [{"token_ids": [3,4,5]}, {"token_ids": [6]}, {"token_ids": [7,8]}, {"token_ids": [9,10,11,2]}]
        adv = advantages([0,1,0,1])[2]
        grads, entropy = [], []
        for micro in (1,2,4):
            model = copy.deepcopy(base)
            c = load_config(ROOT/"configs/experiment.json")["train"]
            c["response_microbatch"] = micro
            metrics = accumulate_group(model, [1,2], responses, adv, c, 0)
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
            entropy.append(metrics["token_entropy_nats"])
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(grads[0], grads[2], rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(entropy[0], entropy[2], places=5)
        # Direct full-group scalar objective, excluding prompt and all padding.
        manual = copy.deepcopy(base)
        losses = []
        for r,a in zip(responses,adv):
            tokens = torch.tensor([[1,2]+r["token_ids"]])
            lp = manual(tokens).logits[0,1:-1].log_softmax(-1)
            losses.append(-a*lp.gather(-1,tokens[0,2:,None]).mean()/4)
        sum(losses).backward()
        expected = torch.cat([p.grad.flatten() for p in manual.parameters()])
        torch.testing.assert_close(grads[0],expected,rtol=1e-5,atol=1e-7)

    def test_rng_eval_isolation_and_restore_on_exception(self):
        import torch
        import numpy as np
        from grpo_order.core import capture_rng, isolated_rng, restore_rng, seed_all
        seed_all(123)
        state = capture_rng()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        restore_rng(state)
        try:
            with isolated_rng(999):
                random.random(); np.random.rand(); torch.rand(30)
                raise ValueError("interrupted evaluation")
        except ValueError:
            pass
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_checkpoint_resume_exact_optimizer_rng_and_cursor(self):
        import torch
        from grpo_order.core import capture_rng, restore_rng, seed_all
        from grpo_order.storage import load_checkpoint, save_checkpoint
        def setup():
            seed_all(7)
            model=torch.nn.Linear(3,1)
            return model,torch.optim.AdamW(model.parameters(),lr=.001)
        def update(model,opt):
            opt.zero_grad()
            x=torch.rand(2,3)
            model(x).square().mean().backward()
            opt.step()
        model,opt=setup()
        for _ in range(6): update(model,opt)
        expected=copy.deepcopy(model.state_dict())
        model,opt=setup()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(3): update(model,opt)
            state={"contract":{"test":1},"cursor":3,"adapter":model.state_dict(),"optimizer":opt.state_dict(),"rng":capture_rng(),"last_record":{"exposure":3}}
            path=save_checkpoint(root,state,[0,3,6])
            with self.assertRaises(ValueError): load_checkpoint(path,{"test":2})
            loaded=load_checkpoint(path,{"test":1})
            restored,restored_opt=setup()
            restored.load_state_dict(loaded["adapter"])
            restored_opt.load_state_dict(loaded["optimizer"])
            restore_rng(loaded["rng"])
            seen=list(range(loaded["cursor"]))
            for i in range(loaded["cursor"],6):
                update(restored,restored_opt); seen.append(i)
            self.assertEqual(seen,list(range(6)))
            for k,v in expected.items(): torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            with self.assertRaises(FileExistsError): save_checkpoint(root,state,[0,3,6])
            with (path/"state.pt").open("ab") as f: f.write(b"corrupt")
            with self.assertRaises(ValueError): load_checkpoint(path,{"test":1})


@unittest.skipUnless(HAS_HF, "HF/PEFT unavailable; install requirements on server")
class HFTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from grpo_order.core import load_config
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path=Path(self.tmp.name)/'model'
        self.c=load_config(ROOT/"configs/experiment.json")
        # Tiny random CPU fixture is deliberately not a production Hub model.
        for spec in self.c['models'].values():
            spec.pop('hub_id',None)
        self.c["train"]["gradient_checkpointing"]=False
        self.c["evaluation"]["max_new_tokens"]=3
        torch.manual_seed(11)
        GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=16,n_layer=1,n_head=2,n_positions=64,
                                 bos_token_id=0,eos_token_id=1,pad_token_id=1)).save_pretrained(self.model_path)

    def test_same_lora_initialization_and_only_lora_updates(self):
        import torch
        from grpo_order.model import load_model, adapter_hash
        from grpo_order.objective import accumulate_group
        self.c['train']['gradient_checkpointing']=True
        a,ma=load_model(self.model_path,self.c,20001,device="cpu",cpu_test=True)
        torch.manual_seed(999)
        b,mb=load_model(self.model_path,self.c,20001,device="cpu",cpu_test=True)
        self.assertEqual(ma["initial_adapter_hash"],mb["initial_adapter_hash"])
        self.assertEqual(ma["targets"],mb["targets"])
        frozen={n:p.detach().clone() for n,p in a.named_parameters() if not p.requires_grad}
        self.assertTrue(all('lora_' in n for n,p in a.named_parameters() if p.requires_grad))
        opt=torch.optim.AdamW([p for p in a.parameters() if p.requires_grad],lr=.01)
        accumulate_group(a,[0,2],[{"token_ids":[3+i,1]} for i in range(4)],[-1,1,-1,1],self.c["train"],1)
        opt.step()
        self.assertNotEqual(adapter_hash(a),adapter_hash(b))
        for n,p in a.named_parameters():
            if n in frozen: torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)

    def test_evaluation_real_generation_does_not_change_rng_weights_or_mode(self):
        import torch
        from grpo_order.core import capture_rng, restore_rng, seed_all
        from grpo_order.model import load_model, adapter_hash
        from grpo_order.evaluation import evaluate
        model,_=load_model(self.model_path,self.c,20001,device="cpu",cpu_test=True)
        model.train()
        class Tok:
            eos_token_id=1; pad_token_id=1; bos_token_id=0
            grpo_eos={"eos_token_ids":[1],"pad_token_id":1,"source":"CPU fixture"}
            def get_vocab(self): return {str(i):i for i in range(32)}
            def encode(self,text,**kwargs): return [0,2,3]
            def decode(self,ids,**kwargs): return str(ids)
        seed_all(123)
        state=capture_rng()
        expected=torch.rand(4)
        restore_rng(state)
        before=adapter_hash(model)
        with patch('grpo_order.evaluation.grade',return_value={"reward":0,"extracted_answer":None,"failure_reason":"fixture"}):
            result=evaluate(model,Tok(),self.c,[{"id":"q","problem":"x","answer":"1","difficulty":"Easy"}],"diagnostic",0,Path(self.tmp.name)/'out',{"test":1})
        self.assertEqual(result["metrics"]["questions"],1)
        self.assertTrue(model.training)
        self.assertEqual(before,adapter_hash(model))
        torch.testing.assert_close(expected,torch.rand(4),rtol=0,atol=0)

    def test_training_loop_interruption_resume_matches_uninterrupted(self):
        import torch
        from grpo_order.core import read_json
        from grpo_order.model import load_model as actual_load_model
        from grpo_order.storage import load_checkpoint, run_path, save_checkpoint as actual_save
        from grpo_order.train import train
        class Tok:
            eos_token_id=1; pad_token_id=1; bos_token_id=0
            grpo_eos={"eos_token_ids":[1],"pad_token_id":1,"source":"CPU fixture"}
            def get_vocab(self): return {str(i):i for i in range(32)}
            def encode(self,text,**kwargs): return [0,2,3]
            def decode(self,ids,**kwargs): return str(ids)
        c=copy.deepcopy(self.c)
        c['train']['max_new_tokens']=2
        c['evaluation']['max_new_tokens']=2
        c['models']['base'].update(path=str(self.model_path),role_verified=True)
        rows=[{"id":f"{d}-{i}","problem":"fixture", "answer":"1","difficulty":d}
              for d in ('Easy','Medium','Hard') for i in range(100)]
        splits={"train":rows,"diagnostic":rows[:3],"validation":rows[:1],"math500":rows[:1]}
        def loader(path,config,seed):
            return actual_load_model(path,config,seed,device='cpu',cpu_test=True)
        def reward(text,answer):
            # Deterministic nonconstant fixture grading, no claim of a math grader test.
            value=int(text.strip('[]').split(',')[0]) % 2
            return {"reward":value,"extracted_answer":str(value),"failure_reason":None if value else 'fixture_wrong'}
        with ExitStack() as stack:
            patches={
                'grpo_order.train.hardware':lambda:{"uuid_mapping":[{"uuid":"GPU-cpu-fixture"}]}, 'grpo_order.train.gpu_errors':lambda *a,**kw:[],
                'grpo_order.train.dependency_errors':lambda:[],
                'grpo_order.train.load_prepared':lambda c:({"hash":"fixture-data"},splits),
                'grpo_order.train.load_tokenizer':lambda *a:(Tok(),64),
                'grpo_order.train.load_model':loader, 'grpo_order.train.gold_parse':lambda x:None,
                'grpo_order.train.grade':reward, 'grpo_order.evaluation.grade':reward,
                'torch.cuda.reset_peak_memory_stats':lambda:None, 'torch.cuda.synchronize':lambda:None,
                'torch.cuda.max_memory_allocated':lambda:0, 'torch.cuda.max_memory_reserved':lambda:0,
                'torch.cuda.mem_get_info':lambda:(0,0)}
            for name,value in patches.items(): stack.enter_context(patch(name,side_effect=value))
            stack.enter_context(patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'}))
            c['output_dir']=str(Path(self.tmp.name)/'uninterrupted')
            train(c,'base','random',1,'smoke')
            aroot=run_path(c,'smoke','base','random',1)
            acontract=read_json(aroot/'metadata.json')['contract']
            expected=load_checkpoint(aroot/'checkpoints/step-000003',acontract)
            c['output_dir']=str(Path(self.tmp.name)/'resumed')
            def interrupt_after_commit(root,state,milestones):
                result=actual_save(root,state,milestones)
                if state['cursor']==1: raise RuntimeError('injected failure after committed update')
                return result
            with patch('grpo_order.train.save_checkpoint',side_effect=interrupt_after_commit):
                with self.assertRaisesRegex(RuntimeError,'injected failure'):
                    train(c,'base','random',1,'smoke')
            train(c,'base','random',1,'smoke',resume=True)
            broot=run_path(c,'smoke','base','random',1)
            bcontract=read_json(broot/'metadata.json')['contract']
            actual=load_checkpoint(broot/'checkpoints/step-000003',bcontract)
            self.assertEqual(actual['optimizer_steps'],3)
            self.assertEqual(actual['generated_responses'],12)
            self.assertEqual(actual['adapter_hash'],expected['adapter_hash'])
            for k,v in expected['adapter'].items(): torch.testing.assert_close(v,actual['adapter'][k],rtol=0,atol=0)
            asteps=[read_json(p) for p in sorted((aroot/'steps').glob('*.json'))]
            bsteps=[read_json(p) for p in sorted((broot/'steps').glob('*.json'))]
            self.assertEqual([r['problem_id'] for r in asteps],[r['problem_id'] for r in bsteps])
            self.assertEqual([r['responses'] for r in asteps],[r['responses'] for r in bsteps])
            self.assertEqual(actual['optimizer']['param_groups'], expected['optimizer']['param_groups'])
            for param, state in expected['optimizer']['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(actual['optimizer']['state'][param][key], value, rtol=0, atol=0)
            torch.testing.assert_close(actual['rng']['torch'],expected['rng']['torch'],rtol=0,atol=0)
            self.assertEqual(actual['rng']['python'], expected['rng']['python'])
            c['output_dir']=str(Path(self.tmp.name)/'stopped')
            paused=train(c,'base','random',1,'smoke',stop_after=1)
            self.assertEqual(paused['stopped_at'],1)
            paused_root=run_path(c,'smoke','base','random',1)
            self.assertFalse((paused_root/'complete.json').exists())
            train(c,'base','random',1,'smoke',resume=True)
            paused_contract=read_json(paused_root/'metadata.json')['contract']
            resumed=load_checkpoint(paused_root/'checkpoints/step-000003',paused_contract)
            self.assertEqual(resumed['adapter_hash'],expected['adapter_hash'])
            # Pilot uses validation only, including a step-0-only prompt pilot.
            c['output_dir']=str(Path(self.tmp.name)/'pilot')
            c['pilot']['updates']=0
            train(c,'base','random',1,'pilot')
            pilot_root=run_path(c,'pilot','base','random',1)
            self.assertTrue((pilot_root/'eval/validation_step-000000.json').is_file())
            self.assertEqual(len(list((pilot_root/'eval').glob('*.json'))),1)

    def test_step0_cache_real_outputs_identity_invalidation_and_rng(self):
        import torch
        from grpo_order.core import capture_rng, restore_rng, seed_all, digest, packages
        from grpo_order.model import load_model
        from grpo_order.evaluation import evaluate
        from grpo_order.step0 import cache_identity
        class Tok:
            eos_token_id=1; pad_token_id=1; bos_token_id=0; chat_template=None
            grpo_eos={'eos_token_ids':[1],'source':'fixture'}
            def get_vocab(self): return {str(i):i for i in range(32)}
            def encode(self,text,**kw): return [0,2,3]
            def decode(self,ids,**kw): return str(ids)
        model,_=load_model(self.model_path,self.c,20001,device='cpu',cpu_test=True)
        self.c['output_dir']=str(Path(self.tmp.name)/'cache')
        rows=[{'id':'q','problem':'x','answer':'1','difficulty':'Easy'}]
        contract={'model':'verified-model-bytes','code':'code','packages':packages(),'mode':'pilot','order':'random'}
        seed_all(17); state=capture_rng(); expected=torch.rand(5); restore_rng(state)
        with patch('grpo_order.evaluation.grade',return_value={'reward':0}):
            a=evaluate(model,Tok(),self.c,rows,'validation',0,Path(self.tmp.name)/'a',contract)
            b=evaluate(model,Tok(),self.c,rows,'validation',0,Path(self.tmp.name)/'b',{**contract,'order':'curriculum'})
        self.assertFalse(a['cache']['hit']); self.assertTrue(b['cache']['hit'])
        self.assertEqual(a['questions'][0]['sampled'],b['questions'][0]['sampled'])
        torch.testing.assert_close(expected,torch.rand(5),rtol=0,atol=0)
        original=cache_identity(model,Tok(),self.c,rows,'validation',contract)
        for field in ('answer','problem','id'):
            modified=[{**rows[0],field:'different'}]
            self.assertNotEqual(digest(original),digest(cache_identity(model,Tok(),self.c,modified,'validation',contract)))
        for section,field,value in [('seeds','evaluation',999),('evaluation','top_p',.9),
                                    ('evaluation','max_new_tokens',2),('prompt','template','new {problem}')]:
            c=copy.deepcopy(self.c); c[section][field]=value
            self.assertNotEqual(digest(original),digest(cache_identity(model,Tok(),c,rows,'validation',contract)))
        other,_=load_model(self.model_path,self.c,20002,device='cpu',cpu_test=True)
        self.assertNotEqual(digest(original),digest(cache_identity(other,Tok(),self.c,rows,'validation',contract)))
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(1)
        with self.assertRaisesRegex(ValueError,'unchanged initial'):
            cache_identity(model,Tok(),self.c,rows,'validation',contract)


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify unavailable")
class GraderTests(unittest.TestCase):
    def test_equivalence_and_extraction_failure(self):
        from grpo_order.grading import grade
        self.assertEqual(grade(r"The answer is $\boxed{\frac{1}{2}}$", "0.5")["reward"],1)
        self.assertEqual(grade("I do not know.","1")["reward"],0)
        self.assertEqual(grade("I do not know.","1")["failure_reason"],"answer_extraction_failed")


if __name__ == "__main__":
    unittest.main()

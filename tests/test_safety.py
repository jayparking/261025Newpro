"""Regression tests for device identity, data audits, generation and cache safety."""
import copy
import json
from pathlib import Path
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grpo_order.core import atomic_json, load_config, read_json
from grpo_order.grading import GoldIssue, GraderEnvironmentError, grader_identity
from test_tensor import HAS_TORCH, HAS_HF

ROOT = Path(__file__).resolve().parents[1]


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c = load_config(ROOT / 'configs/experiment.json')
        self.c['output_dir'] = str(self.root / 'runs')

    def sources(self):
        rng = random.Random(41)
        def row(i, difficulty='Easy', answer='1'):
            return {'id':i, 'difficulty':difficulty, 'answer':answer,
                    'problem':''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=100))}
        splits = {'pool':[row(f'{d}-{i}', d) for d in ('Easy','Medium','Hard') for i in range(4)],
                  'validation':[row('valid'), row('bad-val', answer='unsupported')],
                  'math500':[row(f'm{i}') for i in range(500)]}
        splits['pool'] += [row('bad-pool', answer=''), row('parser-pool', answer='unsupported')]
        self.c['data'].update(prepared_dir=str(self.root/'prepared'), import_manifest=None,
                              require_origin_metadata=False, difficulty_field='difficulty',
                              difficulty_mapping={d:d for d in ('Easy','Medium','Hard')},
                              train_per_difficulty=1, diagnostic_per_difficulty=1)
        for split, rows in splits.items():
            path = self.root / f'{split}.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.c['data'][split] = str(path)
        return splits

    @staticmethod
    def parse(answer):
        if not answer: raise GoldIssue('question_problem', 'missing_or_empty_gold')
        if answer == 'unsupported': raise GoldIssue('parser_limitation', 'no_symbolic_parse')
        return [1]

    def test_prepare_classifies_gold_before_selection(self):
        from grpo_order.data import prepare, load_prepared
        self.sources()
        with patch('grpo_order.data.check_grader',return_value=grader_identity()), patch('grpo_order.data.gold_parse',side_effect=self.parse):
            manifest = prepare(self.c)
        _, splits = load_prepared(self.c)
        self.assertEqual(len(splits['math500']), 500)
        self.assertEqual(manifest['gold_audit']['math500_failed_ids'], [])
        exclusions = read_json(self.root/'prepared/exclusions.json')
        self.assertEqual({e['excluded_id']:e['category'] for e in exclusions}, {
            'bad-pool':'question_problem', 'parser-pool':'parser_limitation', 'bad-val':'parser_limitation'})

    def test_math500_failure_reports_ids_and_never_creates_reduced_split(self):
        from grpo_order.data import prepare
        splits = self.sources()
        splits['math500'][1]['answer'] = 'unsupported'
        Path(self.c['data']['math500']).write_text(''.join(json.dumps(r)+'\n' for r in splits['math500']))
        with patch('grpo_order.data.check_grader',return_value=grader_identity()), patch('grpo_order.data.gold_parse',side_effect=self.parse):
            with self.assertRaisesRegex(ValueError, 'MATH500 gold audit failed'): prepare(self.c)
        self.assertFalse((self.root/'prepared').exists())
        report = read_json(next(self.root.glob('*-gold-audit-*.json')))
        self.assertEqual(report['math500_failed_ids'], ['m1'])
        self.assertEqual(report['math500_count'], 500)

    def test_installation_failure_is_not_a_dataset_exclusion(self):
        from grpo_order.data import prepare
        self.sources()
        with patch('grpo_order.data.check_grader',side_effect=GraderEnvironmentError('missing antlr')):
            with self.assertRaises(GraderEnvironmentError): prepare(self.c)
        self.assertFalse((self.root/'prepared').exists())
        self.assertEqual(read_json(next(self.root.glob('*-gold-audit-*.json')))['category'], 'installation_error')

    def test_cuda_uuid_mapping_ignores_host_indices_and_filters_processes(self):
        from grpo_order.preflight import gpu_errors, uuid_mapping
        h = {'nvidia_smi':{'returncode':0,'stdout':'7, GPU-b, card\n3, GPU-a, card'},
             'processes':{'returncode':0,'stdout':'GPU-b, 999999, other, 100'},
             'devices':[{'logical_index':0,'uuid':'GPU-b','bf16':True,'bf16_kernel_backward':'passed','free_bytes':20*1024**3},
                        {'logical_index':1,'uuid':'GPU-a','bf16':True,'bf16_kernel_backward':'passed','free_bytes':20*1024**3}]}
        self.assertEqual([m['nvidia_smi_index'] for m in uuid_mapping(h)], ['7','3'])
        # Actual pinned torch emits an unprefixed UUID body; nvidia-smi adds GPU-.
        h['devices'][0]['uuid'] = 'b'
        self.assertEqual(uuid_mapping(h)[0]['uuid'], 'GPU-b')
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'0,1'}):
            self.assertTrue(any('compute processes' in e for e in gpu_errors(h, self.c)))
        h['devices'][0]['uuid'] = ''
        with self.assertRaisesRegex(ValueError, 'Cannot match'): uuid_mapping(h)

    def test_freeze_blocks_unfrozen_and_changed_configuration(self):
        from grpo_order.freeze import freeze, verify_freeze
        self.c['evaluation']['length_finalized'] = True
        with self.assertRaisesRegex(ValueError, 'not frozen'): verify_freeze(self.c)
        report = {'errors':[], 'models':{'base':{'content_hash':'base'}, 'rl':{'content_hash':'rl'}}}
        with patch('grpo_order.freeze.load_prepared',return_value=({'hash':'data'},{})), patch('grpo_order.preflight.check',return_value=report):
            freeze(self.c)
            verify_freeze(self.c, report['models'])
            with self.assertRaises(FileExistsError): freeze(self.c)
            with self.assertRaisesRegex(ValueError, 'model/tokenizer'): verify_freeze(self.c, {'base':{'content_hash':'changed'}})
            self.c['train']['lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'changed'): verify_freeze(self.c)

    def test_small_pilots_select_same_set_before_ordering(self):
        from grpo_order.train import select_rows
        from grpo_order.core import seed_plan
        rows = [{'id':f'{d}{i}', 'difficulty':d} for d in ('Easy','Medium','Hard') for i in range(100)]
        for updates in (0, 3, 10, 300):
            self.c['pilot']['updates'] = updates
            a = select_rows({'train':rows}, 'random', seed_plan(self.c,1), 'pilot', self.c)
            b = select_rows({'train':rows}, 'curriculum', seed_plan(self.c,1), 'pilot', self.c)
            self.assertEqual(len(a), updates)
            self.assertEqual({r['id'] for r in a}, {r['id'] for r in b})

    def test_cache_concurrent_writers_and_checksum(self):
        from grpo_order.step0 import shared_evaluate
        calls = []
        def evaluator(model,tok,c,rows,split,step,root,contract):
            calls.append(1)
            time.sleep(.1)
            result = {'contract':contract, 'questions':[{'evaluation_id':'producer'}], 'wall_seconds_this_attempt':1}
            atomic_json(root/'eval/validation_step-000000.json', result)
            return result
        def run(i):
            return shared_evaluate(None,None,self.c,[],'validation',self.root/f'run{i}',{'order':i},evaluator,str(i))
        with patch('grpo_order.step0.cache_identity', return_value={'same':'verified','initial_output_witness':[]}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run, range(2)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(sorted(r['cache']['hit'] for r in results), [False,True])
            self.assertEqual([r['questions'][0]['evaluation_id'] for r in results], ['0','1'])
            cache = Path(results[0]['cache']['source'])
            atomic_json(cache/'eval/validation_step-000000.json', {'corrupted':True})
            with self.assertRaisesRegex(ValueError,'checksum'): run(2)


@unittest.skipUnless(HAS_TORCH and HAS_HF, 'CPU torch/transformers required')
class GenerationSafetyTests(unittest.TestCase):
    def test_resolve_eos_from_actual_special_tokens_and_reject_unverified_ids(self):
        from grpo_order.model import resolve_eos
        class Tok:
            eos_token_id=2; pad_token_id=2; all_special_ids=[2,7]
            def get_vocab(self): return {'eos':2,'<|im_end|>':7,'word':8}
            def convert_ids_to_tokens(self,i): return {2:'eos',7:'<|im_end|>',8:'word'}[i]
            def encode(self,s,**kw): return [self.get_vocab()[s]]
        resolved = resolve_eos(Tok(),SimpleNamespace(eos_token_id=2))
        self.assertEqual(resolved['eos_token_ids'],[2,7])
        with self.assertRaisesRegex(ValueError,'special token'): resolve_eos(Tok(),SimpleNamespace(eos_token_id=8))

    def test_batch_first_eos_trimming_and_budget_eos(self):
        import torch
        from grpo_order.model import generate
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.ones(1)); self.calls=[]
            def generate(self, input_ids, attention_mask, generation_config):
                self.calls.append(generation_config.num_return_sequences)
                return torch.tensor([[9,3,2,2,2],[9,4,5,7,2],[9,4,5,6,7],[9,4,5,6,8]])
        class Tok:
            grpo_eos={'eos_token_ids':[2,7]}; pad_token_id=2; bos_token_id=None
            def decode(self,tokens,**kw): return str(tokens)
        model=Model()
        responses=generate(model,Tok(),[9],count=4,max_new_tokens=4,temperature=1,top_p=1)
        self.assertEqual(model.calls,[4])
        self.assertEqual([r['generated_tokens'] for r in responses],[2,3,4,4])
        self.assertEqual([r['finish_reason'] for r in responses],['eos','eos','eos','length'])
        self.assertTrue(responses[2]['hit_token_budget'])
        self.assertFalse(responses[2]['truncated'])
        self.assertTrue(responses[3]['truncated'])
        self.assertTrue(model.training)

    def test_generation_error_restores_mode_and_both_determinism_flags(self):
        import torch
        from grpo_order.model import generate
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.ones(1))
            def generate(self, **kw):
                self.assert_warn = torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError('injected generation error')
        tok=SimpleNamespace(grpo_eos={'eos_token_ids':[1]}, pad_token_id=1,bos_token_id=0)
        original=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            torch.use_deterministic_algorithms(True,warn_only=False)
            model=Model()
            with self.assertRaisesRegex(RuntimeError,'injected'):
                generate(model,tok,[0],count=8,max_new_tokens=2,temperature=.6,top_p=.95)
            self.assertTrue(model.assert_warn)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(model.training)
        finally:
            torch.use_deterministic_algorithms(original[0],warn_only=original[1])

    def test_zero_advantage_backward_preserves_adam_momentum_updates(self):
        import torch
        from grpo_order.objective import response_loss
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.01,weight_decay=.1)
        p.square().sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        before=p.detach().clone()
        response_loss(p,0.,4,.2).backward()
        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad.item(),0)
        opt.step()
        self.assertFalse(torch.equal(p,before))
        opt.zero_grad(set_to_none=True); before=p.detach().clone(); opt.step()
        torch.testing.assert_close(p,before,rtol=0,atol=0)

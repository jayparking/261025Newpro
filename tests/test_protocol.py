import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from grpo_order.core import (DIFFICULTIES, advantages, atomic_json, digest, load_config,
                             ordered_problems, question_metrics, seed_plan)
from grpo_order.data import load_prepared, normalize, prepare
from grpo_order.analysis import paired_differences, bootstrap_question_means
from grpo_order.storage import exclusive_lock, reconcile_event

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs/experiment.json")
        self.rows = [{"id": f"{d}-{i:03d}", "difficulty": d} for d in DIFFICULTIES for i in range(100)]

    def test_order_no_replacement_no_hidden_shuffle(self):
        for seed in (1, 2, 3):
            a = ordered_problems(self.rows, "random", seed)
            b = ordered_problems(self.rows, "curriculum", seed)
            self.assertEqual({r["id"] for r in a}, {r["id"] for r in b})
            self.assertEqual(len(a), len({r["id"] for r in a}))
            self.assertEqual([r["difficulty"] for r in b], [d for d in DIFFICULTIES for _ in range(100)])
            self.assertEqual(a, ordered_problems(list(reversed(self.rows)), "random", seed))
            self.assertEqual(a[:111] + a[111:], a)  # Exposure cursor defines suffix on resume.

    def test_seed_namespaces_and_fixed_selection(self):
        plans = [seed_plan(self.c, s) for s in (1, 2, 3)]
        self.assertEqual(len({p["data_selection"] for p in plans}), 1)
        self.assertEqual(len({p["evaluation"] for p in plans}), 1)
        self.assertEqual(len({p["lora"] for p in plans}), 3)
        self.assertEqual(len({plans[0][k] for k in ("order", "lora", "generation", "evaluation")}), 4)

    def test_advantage_zero_variance_and_population_std(self):
        for rewards in ([0]*4, [1]*4):
            self.assertEqual(advantages(rewards)[2], [0.0]*4)
        mean, std, adv, kind = advantages([0, 1, 0, 1])
        self.assertEqual((mean, std, kind), (0.5, 0.5, "mixed"))
        self.assertAlmostEqual(sum(adv), 0)
        self.assertAlmostEqual(adv[1], 1.0, places=6)

    def test_avg_pass_and_question_clustered_mixed_groups(self):
        m = question_metrics([1, 0, 0, 0, 0, 0, 0, 0], 0)
        self.assertEqual(m["avg_at_8"], 0.125)
        self.assertEqual(m["pass_at_8"], 1)
        self.assertEqual(m["mixed_fraction"], 0.5)
        self.assertEqual(m["all_wrong_fraction"], 0.5)
        with self.assertRaises(ValueError):
            question_metrics([1]*4, 1)

    def test_paired_analysis(self):
        a = [{"id": "a", "avg_at_8": 0.25}, {"id": "b", "avg_at_8": 0.5}]
        b = [{"id": "b", "avg_at_8": 0.75}, {"id": "a", "avg_at_8": 0.5}]
        d = paired_differences(a, b, "avg_at_8")
        self.assertEqual(d, {"a": 0.25, "b": 0.25})
        self.assertEqual(bootstrap_question_means([d]*3, 1, 20), (0.25, 0.25))
        with self.assertRaises(ValueError):
            paired_differences(a, b[:1], "avg_at_8")

    def test_loss_protocol_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.c["train"]["loss"] = "dapo"
            atomic_json(Path(tmp)/"c.json", self.c)
            with self.assertRaises(ValueError):
                load_config(Path(tmp)/"c.json")

    def test_exclusive_lock_and_event_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"lock"
            with exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(path):
                        pass
            state = {"cursor": 2, "last_record": {"problem_id": "b", "exposure": 2}}
            reconcile_event(tmp, state)
            reconcile_event(tmp, state)
            state["last_record"]["problem_id"] = "wrong"
            with self.assertRaises(ValueError):
                reconcile_event(tmp, state)

    def test_prepared_manifest_fixed_sets_dedup_and_tamper(self):
        self.c['data'].update(difficulty_field='difficulty', difficulty_mapping={d:d for d in DIFFICULTIES},
                              require_origin_metadata=False, import_manifest=None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = random.Random(42)
            def row(i, difficulty):
                text = " ".join(''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12)) for _ in range(10))
                return {"id": str(i), "problem": text, "answer": "1", "difficulty": difficulty}
            pool = [row(f"{d}-{i}", d) for d in DIFFICULTIES for i in range(112)]
            val = [row(f"v{i}", "Easy") for i in range(5)]
            benchmark = [row(f"m{i}", "Easy") for i in range(500)]
            # Exact cross-split duplicate is excluded from the pool, benchmark remains intact.
            pool.append({**benchmark[0], "id": "leak"})
            near = {**pool[0], "id": "near"}
            near["problem"] = near["problem"][:-1] + "!"
            pool.append(near)
            for name, records in (("pool", pool), ("validation", val), ("math500", benchmark)):
                path = root/f"{name}.jsonl"
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                self.c["data"][name] = str(path)
            self.c["data"].update(prepared_dir=str(root/"prepared"), difficulty_rationale="Synthetic test fixture only", source_provenance="test")
            from grpo_order.grading import grader_identity
            with patch('grpo_order.data.check_grader',return_value=grader_identity()), patch('grpo_order.data.gold_parse',return_value=[1]):
                manifest = prepare(self.c)
            _, splits = load_prepared(self.c)
            self.assertEqual(len(splits["train"]), 300)
            self.assertEqual(len(splits["diagnostic"]), 36)
            self.assertEqual(len(splits["math500"]), 500)
            all_ids = [r["id"] for rows in splits.values() for r in rows]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            exclusions = json.loads((root/"prepared/exclusions.json").read_text())
            self.assertEqual(len(exclusions), 2)
            with self.assertRaises(FileExistsError):
                prepare(self.c)
            before = [r["id"] for r in splits["train"]]
            self.c["seeds"]["generation_base"] += 123
            _, again = load_prepared(self.c)
            self.assertEqual(before, [r["id"] for r in again["train"]])
            (root/"prepared/train.json").write_text('[]')
            with self.assertRaises(ValueError):
                load_prepared(self.c)

    def test_text_normalization(self):
        self.assertEqual(normalize("Ａ  B\nC"), normalize("a b c"))

    def test_launcher_twelve_runs_paired_gpu_isolation(self):
        from grpo_order.launcher import launch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            self.c['output_dir']=str(Path(tmp)/'runs')
            self.c['evaluation']['length_finalized']=True
            events=[]
            def preflight(cmd,**kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                self.assertEqual(kwargs['env']['CUDA_DEVICE_ORDER'],'PCI_BUS_ID')
                atomic_json(cmd[cmd.index('--report')+1],{'errors':[], 'models':{}, 'hardware':{'uuid_mapping':[
                    {'allocated_slot':0,'uuid':'GPU-host-seven'}, {'allocated_slot':1,'uuid':'GPU-host-three'}]}})
                return SimpleNamespace(returncode=0)
            class Process:
                def __init__(inner,cmd,**kwargs):
                    role=cmd[cmd.index('--model')+1]; order=cmd[cmd.index('--order')+1]
                    seed=cmd[cmd.index('--seed')+1]
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-host-seven' if order=='random' else 'GPU-host-three')
                    self.assertEqual(kwargs['env']['GRPO_ALLOCATED_SLOT'],'0' if order=='random' else '1')
                    events.append(('start',role,order,seed))
                def wait(inner): events.append(('wait',)); return 0
                def poll(inner): return 0
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'0,1'}), patch('grpo_order.launcher.verify_freeze'), patch('grpo_order.launcher.subprocess.run',side_effect=preflight), patch('grpo_order.launcher.subprocess.Popen',Process):
                launch(self.c,'config.json','full')
            starts=[e for e in events if e[0]=='start']
            self.assertEqual(len(starts),12)
            for i in range(0,len(events),4):
                self.assertEqual([e[0] for e in events[i:i+4]],['start','start','wait','wait'])
                self.assertEqual(events[i][1],events[i+1][1])
                self.assertEqual(events[i][3],events[i+1][3])

    def test_analysis_seed_contrasts_end_to_end(self):
        from grpo_order.analysis import analyze
        from grpo_order.storage import run_path
        from grpo_order.core import read_json
        with tempfile.TemporaryDirectory() as tmp:
            self.c['output_dir']=str(Path(tmp)/'runs')
            for seed in (1,2,3):
                for role in ('base','rl'):
                    for order in ('random','curriculum'):
                        root=run_path(self.c,'full',role,order,seed)
                        contract={'config':digest(self.c),'data':'test','code':'test','packages':{},'model':role}
                        atomic_json(root/'metadata.json',{'contract':contract})
                        atomic_json(root/'complete.json',{'skipped_steps':0})
                        value=.25 if order=='random' else (.5 if role=='base' else .75)
                        questions=[{'id':str(i),**{k:value for k in ('avg_at_8','pass_at_8','greedy')}} for i in range(8)]
                        atomic_json(root/'eval/math500_step-000300.json',{'contract':contract,'metrics':{k:value for k in ('avg_at_8','pass_at_8','greedy')},'questions':questions})
            with patch('builtins.print'):
                result=analyze(self.c,'full',Path(tmp)/'report',plots=False,bootstrap_repeats=40)
            for row in result:
                expected=.5 if row['contrast']=='delta_rl' else .25
                self.assertEqual(row['mean'],expected)
                self.assertEqual(row['seed_sample_sd'],0)
            self.assertEqual(len(read_json(Path(tmp)/'report/analysis.json')['seed_results']),9)


if __name__ == "__main__":
    unittest.main()

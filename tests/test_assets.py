import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from grpo_order.assets import (RECEIPT, inventory, safetensor_dtypes, validate_layout,
                               validate_receipt)
from grpo_order.core import atomic_json, digest, jsonl, load_config
from grpo_order.data import read_source
from grpo_order.import_math import boxed_answer, convert_math_row, import_local
from grpo_order.sources import LEVEL_MAPPING, MATH_ID, MATH500_ID, MODEL_IDS, SUBJECTS

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


def make_receipt(path, repo, kind):
    receipt = {"repo_id": repo, "repo_type": kind, "revision": SHA,
               "tokenizer_revision": SHA if kind == "model" else None,
               "tokenizer_repo_id": repo if kind == "model" else None,
               "files": inventory(path)}
    receipt["receipt_hash"] = digest(receipt)
    atomic_json(Path(path) / RECEIPT, receipt)
    return receipt


def fake_model(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "config.json", {"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28})
    for name in ("tokenizer.json", "tokenizer_config.json"):
        atomic_json(path / name, {})
    header = json.dumps({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)


class SourceTests(unittest.TestCase):
    def test_exact_sources_and_no_math_instruct_substitution(self):
        c = load_config(ROOT / 'configs/experiment.json')
        self.assertEqual({k: v['hub_id'] for k,v in c['models'].items()}, MODEL_IDS)
        self.assertEqual(c['assets']['datasets']['math']['hub_id'], MATH_ID)
        self.assertEqual(c['data']['difficulty_mapping'], LEVEL_MAPPING)
        for spec in c['models'].values(): self.assertTrue(spec['path'].startswith('./models/'))
        with tempfile.TemporaryDirectory() as tmp:
            for wrong in ('Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B'):
                bad = copy.deepcopy(c)
                bad['models']['base']['hub_id'] = wrong
                atomic_json(Path(tmp)/'bad.json', bad)
                with self.assertRaisesRegex(ValueError,'substitution'):
                    load_config(Path(tmp)/'bad.json')

    def test_original_level_stable_id_and_metadata_survive(self):
        for level, difficulty in LEVEL_MAPPING.items():
            row = {'problem':'Solve x + 2 = 3.', 'solution':r'Therefore \boxed{\frac{1}{1}}.',
                   'level':level,'type':'Algebra'}
            a=convert_math_row(row,'algebra','train',SHA,'a.parquet',0)
            b=convert_math_row(row,'algebra','train','b'*40,'b.parquet',999)
            self.assertEqual(a['id'],b['id'])
            self.assertEqual(a['difficulty'],difficulty)
            self.assertEqual(a['answer'],r'\frac{1}{1}')
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'rows.jsonl'
                p.write_text(json.dumps(a)+'\n')
                c=load_config(ROOT/'configs/experiment.json')['data']
                prepared=read_source(p,c,True)[0]
                for key in ('source','source_revision','split','subject','level','difficulty','id','solution'):
                    self.assertEqual(prepared[key],a[key])
                bad={**a,'split':'test'}
                p.write_text(json.dumps(bad)+'\n')
                with self.assertRaisesRegex(ValueError,'test problems'): read_source(p,c,True)

    def test_unknown_level_and_bad_gold_are_explicit(self):
        row=convert_math_row({'problem':'p','level':'unknown','solution':'no final answer'},'geometry','train',SHA,'x',0)
        self.assertIsNone(row['difficulty'])
        self.assertIsNone(row['answer'])
        self.assertEqual(row['conversion_exclusion_reasons'],['unknown_original_level','missing_boxed_gold'])

    def test_nested_and_escaped_boxed_gold(self):
        self.assertEqual(boxed_answer(r'\boxed{1} then \boxed{\frac{2}{3}}'),r'\frac{2}{3}')
        self.assertEqual(boxed_answer(r'\fbox{\{1,2\}}'),r'\{1,2\}')
        with self.assertRaises(ValueError): boxed_answer(r'\boxed{\frac{2}{3}')

    def test_receipt_checks_repo_revision_bytes_and_actual_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_model(tmp)
            self.assertEqual(safetensor_dtypes(tmp),{'F32':2})
            validate_layout(tmp,MODEL_IDS['base'],'model')
            make_receipt(tmp,MODEL_IDS['base'],'model')
            self.assertEqual(validate_receipt(tmp,MODEL_IDS['base'])['revision'],SHA)
            with self.assertRaises(ValueError): validate_receipt(tmp,MODEL_IDS['rl'])
            with self.assertRaises(ValueError): validate_receipt(tmp,MODEL_IDS['base'],'b'*40)
            with (Path(tmp)/'model.safetensors').open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): validate_receipt(tmp,MODEL_IDS['base'])

    def test_missing_subset_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            for subset in SUBJECTS[:-1]:
                p=Path(tmp)/subset; p.mkdir()
                (p/'train-0.parquet').touch(); (p/'test-0.parquet').touch()
            with self.assertRaisesRegex(ValueError,'seven'): validate_layout(tmp,MATH_ID,'dataset')

    @unittest.skipUnless(importlib.util.find_spec('huggingface_hub'), 'Hub library unavailable')
    def test_downloader_pins_revision_before_transfer_and_never_trains(self):
        from grpo_order.assets import fetch_asset
        events=[]
        def snapshot(repo_id,**kwargs):
            if kwargs.get('local_files_only'): raise FileNotFoundError('cache absent')
            self.assertEqual(kwargs['revision'],SHA)
            events.append('download')
            fake_model(kwargs['local_dir'])
            return str(kwargs['local_dir'])
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp, \
             patch('huggingface_hub.snapshot_download',side_effect=snapshot), \
             patch('huggingface_hub.HfApi') as api:
            api.return_value.repo_info.return_value=SimpleNamespace(sha=SHA)
            path=Path(tmp)/'base'
            r=fetch_asset(MODEL_IDS['base'],'model',path,on_revision=lambda sha:events.append('pin'))
            self.assertEqual(events,['pin','download'])
            self.assertEqual(r['revision'],r['tokenizer_revision'])
            fetch_asset(MODEL_IDS['base'],'model',path)
            self.assertEqual(events,['pin','download'])  # Local verified assets bypass transfer.

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'PyArrow unavailable')
    def test_real_parquet_all_subjects_all_splits_conversion(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'math'; bench=root/'benchmark'; bench.mkdir()
            for j,subject in enumerate(SUBJECTS):
                (raw/subject).mkdir(parents=True)
                for split,total in (('train',7500),('test',5000)):
                    n=total//7+int(j<total%7)
                    rows=[{'problem':f'{subject} {split} {i}', 'level':f'Level {i%5+1}',
                           'type':subject,'solution':r'The answer is \boxed{1}.'} for i in range(n)]
                    pq.write_table(pa.Table.from_pylist(rows),raw/subject/f'{split}-00000-of-00001.parquet')
            with (bench/'test.jsonl').open('w') as f:
                for i in range(500):
                    f.write(json.dumps({'unique_id':f'test/algebra/{i}.json','problem':f'benchmark {i}',
                                       'answer':'1','level':i%5+1,'subject':'Algebra'})+'\n')
            make_receipt(raw,MATH_ID,'dataset'); make_receipt(bench,MATH500_ID,'dataset')
            c=load_config(ROOT/'configs/experiment.json')
            c['assets']['datasets']['math']['path']=str(raw)
            c['assets']['datasets']['math500']['path']=str(bench)
            c['assets']['converted_dir']=str(root/'converted')
            report=import_local(c)
            self.assertEqual(len(jsonl(root/'converted/all_train.jsonl')),7500)
            self.assertEqual(len(jsonl(root/'converted/all_test.jsonl')),5000)
            self.assertEqual(set(report['subject_counts']),set(SUBJECTS))
            pool=jsonl(root/'converted/pool.jsonl'); val=jsonl(root/'converted/validation.jsonl')
            self.assertEqual(len(val),150)
            self.assertTrue(all(r['split']=='train' for r in pool+val))
            self.assertFalse({r['id'] for r in pool}&{r['id'] for r in val})
            self.assertEqual(len(jsonl(root/'converted/math500.jsonl')),500)
            with self.assertRaises(FileExistsError): import_local(c)


if __name__=='__main__':
    unittest.main()

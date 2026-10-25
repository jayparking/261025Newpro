"""The server entry point must stop on failures and never bypass selection/freeze."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from grpo_order.core import atomic_json, digest, load_config, read_json
from grpo_order.storage import run_path
from scripts.first_pipeline import Pipeline, immutable_json, pilot_summary, session_config, verify_smoke

ROOT = Path(__file__).resolve().parents[1]


class FirstTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = load_config(ROOT/'configs/experiment.json')
        self.c = session_config(self.source, self.root)
        self.state = {'completed':{}, 'choices':{}}
        self.pipeline = Pipeline(self.root, self.state)

    def test_new_session_preserves_original_assets_and_separates_results(self):
        self.assertEqual(self.c['models'], self.source['models'])
        self.assertEqual(self.c['assets']['datasets'], self.source['assets']['datasets'])
        self.assertNotEqual(self.c['data']['prepared_dir'], self.source['data']['prepared_dir'])
        self.assertNotEqual(self.c['assets']['converted_dir'], self.source['assets']['converted_dir'])
        self.assertEqual(self.c['train']['lr'], 5e-6)
        path = immutable_json(self.root/'config.json', self.c)
        immutable_json(path, self.c)
        with self.assertRaisesRegex(ValueError, 'different content'):
            immutable_json(path, self.source)
        self.assertEqual(read_json(path), self.c)

    def test_success_marker_failure_and_resume_do_not_repeat_stage(self):
        output = self.root/'one.txt'
        command = [sys.executable, '-c', 'import sys; from pathlib import Path; Path(sys.argv[1]).open("x").write("one")', str(output)]
        with patch('builtins.print'):
            self.pipeline.run('once', command, once=True)
            self.pipeline.run('once', command, once=True)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                self.pipeline.run('failure', [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(output.read_text(), 'one')
        self.assertNotIn('failure', self.state['completed'])
        self.assertEqual(len(list((self.root/'logs').glob('*.log'))), 2)

    def test_selection_is_bound_to_validation_evidence(self):
        evidence = {'plain':.2,'chat':.3}
        self.state['choices']['prompt'] = {'value':'chat', 'evidence':digest(evidence)}
        self.assertEqual(self.pipeline.choose('prompt','prompt',['plain','chat'],evidence), 'chat')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.pipeline.choose('prompt','prompt',['plain','chat'],{'plain':.5,'chat':.3})

    def test_terminal_choice_records_only_valid_input_and_eof_blocks(self):
        import io
        class Stream(io.StringIO):
            def close(self): pass
        reader = Stream('invalid\nchat\n')
        real_open = open
        def terminal_open(path,mode='r',*args,**kwargs):
            if path == '/dev/tty':
                self.assertEqual(mode,'r')
                return reader
            return real_open(path,mode,*args,**kwargs)
        with patch('builtins.open',side_effect=terminal_open), patch('builtins.print'):
            choice=self.pipeline.choose('prompt','Prompt',['plain','chat'],{'validation':1})
            self.assertEqual(choice,'chat')
            with self.assertRaisesRegex(RuntimeError,'터미널'):
                self.pipeline.choose('length','Length',['2048','4096'],{'validation':2})
        self.assertEqual(read_json(self.root/'state.json')['choices']['prompt']['value'],'chat')

    def test_pipeline_order_and_full_only_after_selection_and_freeze(self):
        for pilot_only, candidate in ((True,'none'), (False,'none'), (False,'1e-5')):
            with self.subTest(pilot_only=pilot_only,candidate=candidate), tempfile.TemporaryDirectory() as tmp:
                session = Path(tmp)
                c = session_config(self.source, session)
                events = []
                class Fake(Pipeline):
                    def pin_assets(self, c): pass
                    def run(self, label, args, **kw): events.append(label)
                    def cli(self, label, command, config, *args, **kw): events.append(label)
                    def pilot(self, name, config, path):
                        self.assert_config = config
                        events.append(name)
                        return {'validation':name}
                    def choose(self, key, title, choices, evidence, default=None):
                        events.append('choose-'+key)
                        return {'prompt':'chat','length':'4096','lr_candidate':candidate,'lr':candidate}[key]
                pipe = Fake(session, {'completed':{},'choices':{}})
                with patch('scripts.first_pipeline.verify_import'), patch('scripts.first_pipeline.load_prepared'), \
                     patch('scripts.first_pipeline.verify_smoke'), patch('builtins.print'):
                    pipe.execute(c, session/'config.json', pilot_only)
                expected = ['environment-lock','cpu-tests','environment','download','import','prepare','preflight',
                            'smoke-stop','smoke-resume','prompt_plain','prompt_chat','choose-prompt',
                            'length_2048','length_4096','choose-length','lr_5e-6','choose-lr_candidate']
                if candidate != 'none':
                    expected += ['lr_'+candidate,'choose-lr']
                if not pilot_only:
                    expected += ['final-preflight','freeze','full-12-runs','analyze']
                self.assertEqual(events, expected)
                final = read_json(session/'configs/final.json')
                self.assertEqual(final['prompt']['mode'], 'chat')
                self.assertEqual(final['evaluation']['max_new_tokens'], 4096)
                self.assertEqual(final['train']['lr'], 5e-6 if candidate == 'none' else float(candidate))
                self.assertTrue(final['evaluation']['length_finalized'])
                self.assertEqual((session/'finished.json').exists(), not pilot_only)

    def test_failed_cpu_gate_cannot_reach_download_or_training(self):
        from scripts.run_tests import main as test_main
        from types import SimpleNamespace
        result=SimpleNamespace(skipped=[('grader','dependency missing')],wasSuccessful=lambda:True)
        with patch('scripts.run_tests.unittest.defaultTestLoader.discover'), \
             patch('scripts.run_tests.unittest.TextTestRunner') as runner, patch('builtins.print'):
            runner.return_value.run.return_value=result
            with patch.object(sys,'argv',['run_tests.py','--require-no-skips']):
                self.assertEqual(test_main(),1)
            with patch.object(sys,'argv',['run_tests.py']):
                self.assertEqual(test_main(),0)
        events=[]
        def fail(label, args, **kw):
            events.append(label)
            if label == 'cpu-tests': raise RuntimeError('skip gate failed')
        with patch.object(self.pipeline, 'run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'skip gate'):
                self.pipeline.execute(self.c,self.root/'config.json',False)
        self.assertEqual(events,['environment-lock','cpu-tests'])

    def test_validation_summary_does_not_read_math500(self):
        self.c['pilot']['updates']=0
        for role in ('base','rl'):
            for order in ('random','curriculum'):
                root=run_path(self.c,'pilot',role,order,1)
                contract={'config':digest(self.c)}
                atomic_json(root/'metadata.json',{'contract':contract})
                atomic_json(root/'complete.json',{'contract':contract,'skipped_steps':0,'cursor':0,'group_counts':{}})
                metrics={key:.25 for key in ('avg_at_8','pass_at_8','greedy','sampled_truncation_fraction','sampled_mean_tokens')}
                atomic_json(root/'eval/validation_step-000000.json',{'contract':contract,'metrics':metrics,
                            'questions':[{'sampled':[{'failure_reason':None}]*8}]})
                (root/'eval/math500_step-000000.json').write_text('this must never be parsed')
        summary=pilot_summary(self.c)
        self.assertEqual(summary['split'],'validation')
        self.assertEqual(summary['mean_over_four_conditions']['avg_at_8'],.25)
        self.assertEqual(summary['mean_over_four_conditions']['delta_avg_at_8'],0)
        self.assertEqual(len(summary['records']),4)

    def test_smoke_must_have_all_updates_and_reload_verification(self):
        expected=copy.deepcopy(self.c)
        expected['train']['max_new_tokens']=expected['evaluation']['max_new_tokens']=32
        completion={'contract':{'config':digest(expected)},'cursor':3,'optimizer_steps':3,
                    'skipped_steps':0,'generated_responses':12,'checkpoint_reload_evaluation':'passed'}
        for role in ('base','rl'):
            for order in ('random','curriculum'):
                atomic_json(run_path(self.c,'smoke',role,order,1)/'complete.json',completion)
        verify_smoke(self.c)
        atomic_json(run_path(self.c,'smoke','base','random',1)/'complete.json',
                    {**completion,'skipped_steps':1,'optimizer_steps':2})
        with self.assertRaisesRegex(ValueError, 'Smoke did not verify'): verify_smoke(self.c)

    def test_shell_help_and_dry_run_work_without_gpu(self):
        for flag in ('--help','--dry-run'):
            p=subprocess.run(['bash',str(ROOT/'first.sh'),flag],cwd=self.root,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('12회',p.stdout)
        p=subprocess.run(['bash',str(ROOT/'first.sh'),'--unknown'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)

    def test_shell_install_and_dispatch_with_linux_command_fixtures(self):
        # Run the real Bash entry point against local command fixtures. No pip,
        # network, GPU or production pipeline is started by this test.
        for install, assigned in ((True,None),(False,'GPU-seven,GPU-three')):
            with self.subTest(install=install), tempfile.TemporaryDirectory(prefix='first shell ') as tmp:
                root=Path(tmp)
                (root/'first.sh').write_text((ROOT/'first.sh').read_text())
                binpath=root/'bin'; binpath.mkdir()
                for name,body in {'uname':'#!/bin/sh\necho Linux\n','flock':'#!/bin/sh\nexit 0\n'}.items():
                    path=binpath/name; path.write_text(body); path.chmod(0o755)
                fake=binpath/'python-fixture'
                fake.write_text('#!'+sys.executable+'\n'+'''import json, os, sys
from pathlib import Path
with open(os.environ['FIRST_TEST_LOG'], 'a') as f:
    f.write(json.dumps({'args':sys.argv[1:], 'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')
if sys.argv[1:3] == ['scripts/bootstrap_venv.py','--venv']:
    dst=Path(sys.argv[3])/'bin'/'python'; dst.parent.mkdir(parents=True)
    dst.write_text(Path(__file__).read_text()); dst.chmod(0o755)
''')
                fake.chmod(0o755)
                env={**os.environ,'PATH':str(binpath)+os.pathsep+os.environ['PATH'],
                     'FIRST_TEST_LOG':str(root/'commands.jsonl'),'PYTHON_BOOTSTRAP':str(fake),'PYTHON':str(fake)}
                env.pop('CUDA_VISIBLE_DEVICES',None)
                if assigned: env['CUDA_VISIBLE_DEVICES']=assigned
                command=['bash',str(root/'first.sh')]+([] if install else ['--skip-install','--pilot-only'])
                result=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                calls=[json.loads(line) for line in (root/'commands.jsonl').read_text().splitlines()]
                self.assertEqual(calls[-1]['args'],['scripts/first_pipeline.py']+([] if install else ['--pilot-only']))
                self.assertEqual(calls[-1]['cuda'],assigned or '0,1')
                pip=[c['args'] for c in calls if c['args'][:2]==['-m','pip']]
                self.assertEqual(len(pip),2 if install else 0)
                if install:
                    self.assertIn('https://download.pytorch.org/whl/cu128',pip[0])
                    self.assertIn('requirements-assets.txt',pip[1])

    def test_source_or_receipt_change_blocks_existing_session(self):
        from grpo_order.core import code_hash, packages
        self.state.update(code=code_hash(),packages=packages())
        self.pipeline.check_identity()
        with patch('scripts.first_pipeline.code_hash',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Code/packages'): self.pipeline.check_identity()
        with patch('scripts.first_pipeline.file_hash',return_value='receipt-original'):
            self.pipeline.pin_assets(self.c)
        with patch('scripts.first_pipeline.file_hash',return_value='receipt-changed'):
            with self.assertRaisesRegex(ValueError,'receipts changed'): self.pipeline.pin_assets(self.c)

    def test_interruption_terminates_child_process_group_and_never_marks_complete(self):
        from types import SimpleNamespace
        import signal
        process=SimpleNamespace(pid=123456789,wait=unittest.mock.Mock(side_effect=[KeyboardInterrupt(),-15]))
        with patch('scripts.first_pipeline.subprocess.Popen',return_value=process) as popen, \
             patch('scripts.first_pipeline.os.killpg') as kill, patch('builtins.print'):
            with self.assertRaises(KeyboardInterrupt):
                self.pipeline.run('interrupted',['fixture'])
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(process.pid,signal.SIGTERM)
        self.assertNotIn('interrupted',self.state['completed'])

    def test_smoke_failure_prevents_every_pilot_and_full(self):
        events=[]
        with patch.object(self.pipeline,'run',side_effect=lambda label,*a,**kw:events.append(label)), \
             patch.object(self.pipeline,'pin_assets'), patch('scripts.first_pipeline.verify_import'), \
             patch('scripts.first_pipeline.load_prepared'), \
             patch('scripts.first_pipeline.verify_smoke',side_effect=ValueError('bad smoke')):
            with self.assertRaisesRegex(ValueError,'bad smoke'):
                self.pipeline.execute(self.c,self.root/'config.json',False)
        self.assertEqual(events[-1],'smoke-resume')
        self.assertFalse(any(label.startswith(('prompt_','length_','lr_','full')) for label in events))

    def test_cpu_test_child_isolated_from_server_gpu_assignment(self):
        from types import SimpleNamespace
        process=SimpleNamespace(wait=lambda:0)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-seven,GPU-three'}), \
             patch('scripts.first_pipeline.subprocess.Popen',return_value=process) as popen, patch('builtins.print'):
            self.pipeline.run('cpu-tests',['fixture'])
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')


if __name__ == '__main__':
    unittest.main()

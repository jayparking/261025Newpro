"""Create an explicit immutable experiment variant, including a separate output root."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("--source",default="configs/experiment.json")
p.add_argument("--output",required=True)
p.add_argument("--train-max-new-tokens",type=int,choices=(2048,4096))
p.add_argument("--eval-max-new-tokens",type=int,choices=(2048,4096))
p.add_argument("--prompt-mode", choices=("plain", "chat"))
p.add_argument("--lr", type=float)
p.add_argument("--pilot-updates", type=int)
p.add_argument("--finalize-length", action="store_true")
p.add_argument("--output-dir",required=True)
a=p.parse_args()
dst=Path(a.output)
if dst.exists(): raise SystemExit(f"Refusing to overwrite {dst}")
c=json.loads(Path(a.source).read_text())
if a.train_max_new_tokens is not None: c['train']['max_new_tokens']=a.train_max_new_tokens
if a.eval_max_new_tokens is not None: c['evaluation']['max_new_tokens']=a.eval_max_new_tokens
if a.prompt_mode: c['prompt']['mode']=a.prompt_mode
if a.lr is not None:
    if not 0 < a.lr < float('inf'): p.error('--lr must be finite and positive')
    c['train']['lr']=a.lr
if a.pilot_updates is not None:
    if not 0 <= a.pilot_updates <= 300: p.error('--pilot-updates must be 0..300')
    c['pilot']['updates']=a.pilot_updates
c['evaluation']['length_finalized']=a.finalize_length
c['output_dir']=a.output_dir
if Path(a.output_dir).exists(): raise SystemExit(f'Refusing existing output directory {a.output_dir}')
dst.parent.mkdir(parents=True,exist_ok=True)
with dst.open('x') as f: json.dump(c,f,indent=2,ensure_ascii=False); f.write('\n')
print(dst)

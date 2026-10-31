#!/usr/bin/env python
"""Print the 00_precision_ladder.py flags that reproduce an existing ladder run, so new
rungs can be added with --resume without re-running anything (CPU only, no imports).

    python scripts/q1_resume_args.py runs/gate2/ladder.json
    -> --n-pope-images 100 --n-chair-images 300 --max-new-tokens 128 ...

Also lists (on stderr) the rungs already in the file, so an existing LoRAS-T rung
can be reused under exactly the same name.
"""
import json
import shlex
import sys

FLAGS = ["n_pope_images", "n_chair_images", "max_new_tokens", "pope_splits", "seed",
         "group_size", "quant_targets", "n_blind_captions", "n_boot", "eval_subset"]

cfg = json.load(open(sys.argv[1]))
table = cfg.get("table", {})
cfg = cfg["config"]
out = []
for k in FLAGS:
    if k in cfg and cfg[k] is not None:
        out += [f"--{k.replace('_', '-')}", str(cfg[k])]
print(" ".join(shlex.quote(x) for x in out))
print(f"[q1] {sys.argv[1]}: {len(table)} rungs, model {cfg.get('model_id')}, dtype {cfg.get('dtype')}, "
      f"{cfg.get('n_chair_images')} captions, {cfg.get('n_pope_images')} POPE images", file=sys.stderr)
for r in table:
    print(f"[q1]   rung: {r}", file=sys.stderr)

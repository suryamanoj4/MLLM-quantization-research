# Side experiments

These are **standalone** follow-up experiments. Each one contains its own `phoenix`
package because it changes core model/quantization/steering code; the main
`method-study/scripts/` pipeline is intentionally left untouched.

| Experiment | Question | Entry point | Outputs |
|---|---|---|---|
| [`patching_qwen/`](patching_qwen/) | Where in the network does quantization lose grounding — layer, component and token positions? | `bash run.sh` | `runs/patching/...` |
| [`q1_lorast_positions/`](q1_lorast_positions/) | Does LoRAS-T work because it corrects **text positions**, or just because it adds low-rank capacity? | `bash run.sh` | `results_q1_<tag>/` |

Both experiments reuse the shared environment and data from the main study:

```bash
cd method-study
bash scripts/setup_env.sh
python scripts/fetch_data.py --root data/coco-mini --preset dev
```

Run them from their own directories. Defaults point at `../../data/coco-mini` and the
shared `../../runs/...` ladders; new artifacts stay inside each experiment.

## Why separate copies?

The zips modify shared files (`model.py`, `acab.py`, `quant.py`, `loras_text.py`,
`12_calibrate_lorast.py`) in ways that are specific to each experiment. Keeping a
complete, runnable copy per experiment:

- makes each experiment self-contained and reproducible on its own;
- prevents experimental changes from silently altering the main LLaVA/Qwen results;
- keeps the import path (`phoenix` next to `scripts/`) identical to the main project.

The trade-off is some duplicated support code. That is deliberate here; these are
experimental branches, not new defaults for the main pipeline.

The original zips are kept locally in `_sources/` for provenance and are gitignored;
the extracted directories above are the maintained code.

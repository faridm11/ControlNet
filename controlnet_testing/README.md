# Where everything is

All experiments of the project live in this folder. The project code is one level up in `DDPM/`; its `README.md` explains
the method, setup, training, evaluation and the downstream test.

## The three documents

| File | What it is for |
|---|---|
| `RESULTS.docx` | the results used in the report: tables, curves, sample images (rebuilt by `reporting/build_results_doc.py`) |
| `EXPERIMENT_LOG.md` | every step in order: what, why, settings, job ids, results; the **FULL FINAL FLOW** section is at the end |
| `RUN_INDEX.md` | every valid training run: folder, settings, best numbers, kept files (rebuilt by `reporting/index_runs.py`) |

## Result folders

| Folder | Contents |
|---|---|
| `FINAL/` | v1 models of both datasets: `run/`, `eval_mapillary/` (three seeds, Mapillary judge), `comparison/` |
| `FINAL_v2/` | v2 = model **B** (fine-tuned, no segmentation loss), both datasets |
| `FINAL_v2_segloss/sanpo/` | B + segmentation loss, λ 0 → 0.006 |
| `FINAL_v2_segloss/sanpo_l0.02/` | B + segmentation loss, λ 0.02 |
| `FINAL_v2_segloss/sanpo_l0.025-0.045/` | **selected model**: λ rising 0.025 → 0.045 |
| `FINAL_v2_segloss/sanpo_ramp/` | λ rising 0.05 → 0.4 (strongest; least realistic) |
| `step18_second_judge/` | independent ADE20K judge; every generated test image per model; low-white / high-white split |
| `step19_judge_mapping/` | check of the judge class mapping (v1 kept) |
| `step20_downstream_sanpo/` | downstream test: 300-photo test, full final flow (200 photos), ImageNet-start variant; `results/` |
| `step11_hparam_screen/` | rank 32 / larger LoRA / flip screen on dataset 1 |

Each run folder has `run/` (logs, plots, samples, checkpoints), `eval/` (three-seed JSONs) and `comparison/` (sample sheet).
Run ids are in every file name: `metrics_<run>.csv`, `<run>_overview.png`, `<run>_LoRA.pt`; SANPO runs start with `sanpo_`.

## Code checks and tools

| Folder | Contents |
|---|---|
| `step0_fixes/`, `step1_cpu_checks/`, `step9_sanpo_audit/`, `step12_review_fixes/`, `step15_phase2_sanpo/`, `step16_lambda_ramp/` | CPU tests of the code fixes and new features |
| `jobs/` | SLURM scripts: `train.sh`, `baseline_eval.sh`, `comparison_grid.sh` |
| `reporting/` | `build_results_doc.py`, `index_runs.py`, `make_comparison_grid.py`, `make_white_gallery.py` |
| `_code_backups/` | copies of source files before each change |

Superseded and bug-affected runs (pre-audit pipeline, the old judge, the dataset-1 runs with the random-state bug) were
moved to `DDPM/_archive_not_in_report/`; its README says why each one is there. Their history stays in `EXPERIMENT_LOG.md`.

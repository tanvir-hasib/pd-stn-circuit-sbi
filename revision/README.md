# Revision analyses (v5.1-rev1)

Analyses added during the revision of the manuscript. All of them run on the **frozen v5.1 pipeline**: each
notebook executes `notebooks/pd_stn_circuit_sbi.py` with its caches (the simulation banks, posterior networks and
classifiers of the v5.1 run are loaded, not retrained), then runs its analyses. New simulations are made only for
calibration and misspecification tests. Nothing here changes the model, the priors, the statistics or the networks.

## Running

Run the main v5.1 notebook first (or have its `OUT_DIR` with caches). Then, from this folder, with the same
environment variables as the main run:

```
DS004998_ROOT=/path/to/ds004998
PD_PAC_OUT=/path/to/pd_pac_v4_outputs          # the SAME output folder as the v5.1 run (caches)
PD_PAC_GPU=0
PD_PAC_PIPELINE=../notebooks/pd_stn_circuit_sbi.py   # default
```

run the notebooks in this order (each later notebook reuses files written by the earlier ones):

| # | notebook | analyses | runtime (RTX 2080 Ti) | needs |
|---|---|---|---|---|
| 1 | `revision_analyses.ipynb` | R1–R9 | ≈ 3–4 h | v5.1 caches |
| 2 | `revision_analyses_2.ipynb` | R10–R12 | + 1.5–2 h | 1 (`revision_analyses.py` in this folder) |
| 3 | `revision_analyses_R12b.ipynb` | R12b | + 25–35 min | 2 (`R12_pairs_*.npz`) |
| 4 | `revision_analyses_R13.ipynb` | R13 A–D | + 4–5 h | 2, 3 |
| 5 | `revision_analyses_R14.ipynb` | R14 | + 2.5–3 h | 4 (`R13/C_cohorts_all.csv`) |

"+": in addition to loading the pipeline with caches (≈ 1.5–2 h). Notebooks 2–5 execute `revision_analyses.py` with
all of its analyses switched off to load the pipeline and the helper functions. Every simulation step saves its
results and resumes from them. Tables go to `OUT_DIR/revision/` (R13/, R14/ subfolders), figures to
`OUT_DIR/revision/figures/`; copy the shareable ones with
`python ../tools/collect_results.py --out-dir $PD_PAC_OUT --dest ../results --revision`.
The `.py` files are the same notebooks in percent format (`# %%` cells).

`executed/` holds the reported runs with outputs. They were run on Windows with local paths, which were replaced by
`<local path>`; Oxford-derived outputs (R7, and the pipeline printout of the Oxford section) were removed with
`tools/strip_restricted_outputs.py`.

## Where each result appears in the manuscript

| Analysis | Content | Manuscript |
|---|---|---|
| R1 | patient-level OFF/ON tests, mixed models, FDR; hierarchical posterior of every candidate change | Table 5, §3.1; §3.3, Fig. S4 |
| R2 | HFO sensitivity to notches, HPI lines, EMG and transients; resampling gain | §3.1, Table S1, Fig. S3 |
| R3 | model-free β–HFO dissociation; PAC partial correlations and mixed model | §3.1 |
| R4 | sensitivity to the sparsity of the change prior | §3.4 |
| R5 | misspecified fixed constants: recovery and false decreases of w_GS | §3.4, Fig. S5 |
| R5b | simulation-calibrated null and alternative cohorts | §3.4, Table 8 |
| R6 | posterior predictive β change under a less sparse prior | §3.4 |
| R7 | Oxford, bipolar channels only (outputs restricted) | §3.8 |
| R8 | channel-selection sensitivity (β-best STN pair) | §3.4 |
| R9 | clinical associations; simulation partitions | §3.1, §3.3; §2.12 |
| R10–R11 | recovery and sensitivity of the hierarchical estimate (R10 superseded by R13 C) | §3.4, Table S2 |
| R12, R12b | complete posterior-signature comparison of scenarios, with w_GS + noise | §3.5, Table 9, Fig. 8 |
| R13 A | hemisphere flow; the out-of-reach pair | §3.1, Table 4; §3.3 |
| R13 B | all joint-network results on the 36 in-reach hemispheres | §3.3–3.4, Tables 6 and 8 |
| R13 C | calibration of the hierarchical estimator on 184 cohorts; bias-corrected ×0.76 [0.71, 0.82] | §3.4, Fig. S6 |
| R13 D | stability of the signature comparison with 12 pairs per hemisphere | §3.5, Table 9 |
| R14 | calibration under alternative true effects (noise only, w_GS + noise, misspecified constants) | §3.4, Table 7, Fig. S7 |

Table and figure numbers refer to the submitted manuscript.

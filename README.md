# Levodopa and the STN–GPe circuit: simulation-based inference on STN-LFP and MEG in Parkinson's disease

A cortex–STN–GPe circuit model with an STN high-frequency-oscillation (HFO) generator, fitted to resting
subthalamic local field potentials (STN-LFP) and MEG recorded OFF and ON levodopa, using amortised
simulation-based inference (neural posterior estimation, amortised model comparison and a joint OFF/ON
posterior network). The question: **which circuit parameters does levodopa change, and do they explain the
medication effects on STN β, HFO and β–HFO phase–amplitude coupling (PAC)?**

Author: Tanvir Hasib (Multimedia University, Malaysia). Analysis version **v5.1** (frozen); model
history in [`CHANGELOG.md`](CHANGELOG.md).

## Main results (v5.1 full run)

ds004998 (Rassoulou et al., 2024): 20 patients, 36–37 OFF/ON hemisphere pairs; Oxford MRC BNDU: 11 OFF/ON pairs
(pre-stimulation segments).

| Finding | Evidence |
|---|---|
| **Levodopa lowers effective GPe→STN coupling by ≈ 14 %** | joint OFF/ON posterior, cohort mean w_GS ×0.86 [90 % CI ×0.80, ×0.93], P(decrease) ≈ 1.00; STN-only network ×0.88, P = 0.997; Oxford (STN-only, n = 10) ×0.91, P = 0.91 |
| The β route is GPe→STN, not slow burst input or noise | family-level RFX-BMS: GPe–STN exceedance probability 1.00 (expected frequency 0.72); σ_slow and noise changes ≈ 0 in the joint posterior |
| The method can tell the routes apart | held-out simulations: AUC 0.91 for a w_GS decrease; attribution on hypothesis pairs is diagonal (0.56–0.68 vs ≈ 0.2 background) |
| **The circuit change explains the β reduction** | posterior predictive 90 % coverage of the observed OFF→ON β change 0.83–0.92 (two sets of draws) vs 0.50 with the candidate changes ablated; w_GS is the dominant driver of Δβ near the patients' OFF states (drop-one R² 0.30) |
| **The HFO increase ON is real but dissociable from the β change** | HFO excess ↑ (p = .029), fast/slow HFO ratio ↑ (p < .001); the β-state HFO gate does not explain it (gated vs frozen gate: no difference, p = .09; gate strength not constrained by the data); observed HFO increases exceed the model's for the same β drop |
| β–HFO PAC is not an independent marker | weak, present in a minority of hemispheres, not medication-dependent, and tracks β and HFO power in both cohorts |

Limitations stated in the analysis: the predicted β change is attenuated relative to the observed one (Bayesian
shrinkage under the sparse change prior; coverage is nominal); κ_H and ν_H are not identifiable; independent
single-session fits do not resolve the w_GS change (the joint network is the within-hemisphere analysis);
the Oxford OFF/ON contrast is underpowered; Δw_GS does not correlate with clinical improvement; the STN–cortex
delay is poorly fitted.

## Repository layout

```
notebooks/pd_stn_circuit_sbi.ipynb   the full analysis (Jupyter)
notebooks/pd_stn_circuit_sbi.py      the same, as a percent-format script (# %% cells)
notebooks/pd_stn_circuit_sbi_executed.ipynb   the v5.1 full run with outputs (Oxford cell outputs removed)
data/README.md                       how to obtain the two datasets
results/                             tables and figures from the full run (see results/README.md)
tools/collect_results.py             copies the shareable outputs from OUT_DIR into results/
tools/strip_restricted_outputs.py    removes Oxford-derived cell outputs from an executed notebook
CHANGELOG.md                         model and analysis history (v4 → v5.1)
requirements.txt, LICENSE, CITATION.cff
```

## Running it

1. **Environment.** Python ≥ 3.10, an NVIDIA GPU with ≥ 11 GB memory, ≈ 8 GB free RAM per simulation
   batch (`BATCH = 2048`; lower it on smaller machines). Install a CUDA build of PyTorch first
   (<https://pytorch.org>), then `pip install -r requirements.txt`.
2. **Data.** See [`data/README.md`](data/README.md). Point the notebook at the data with environment
   variables (or edit §1):
   ```
   DS004998_ROOT=/path/to/ds004998
   OXFORD_BNDU_ROOT=/path/to/oxford_mrc_bndu     # optional; set RUN_OXFORD = False without it
   PD_PAC_GPU=0                                  # CUDA device index
   PD_PAC_OUT=pd_pac_v4_outputs                  # caches and outputs
   ```
3. **Check run.** Leave `QUICK = True` and run all cells (≈ 1 h). It exercises every section on small banks.
4. **Full run.** Set `QUICK = False` and keep the same `OUT_DIR` (≈ 11–12 h on an RTX 2080 Ti):

   | step | time |
   |---|---|
   | simulation bank (49k simulations) | ≈ 5 h |
   | 12-hypothesis OFF/ON pairs (16k) | ≈ 2.5 h |
   | joint OFF/ON bank (24.6k ON partners) | ≈ 2.5 h |
   | posterior networks, classifiers, posterior predictive checks, gate ablation | ≈ 1.5 h |

   Everything is cached in `OUT_DIR`, keyed by a configuration hash; re-running loads the caches.
5. **Collect results:** `python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results`.

## Reproducibility

* Fixed seeds throughout (`SEED = 20261113`); simulation banks, networks and classifiers are cached under a
  hash of the model configuration, prior, statistics and integrator.
* §9 writes a pipeline lock (`pipeline_lock_v51.json`): a hash of the configuration and the source of the
  preprocessing, statistics and simulator functions, taken before the model is fitted to data.
* The CUDA-graph simulator is checked against eager execution at start-up (§2).
* Exact package versions of the reported run: see `requirements.txt`.

## Data and privacy

No patient data are included. ds004998 is public on OpenNeuro. The Oxford MRC BNDU recordings are available
on request from the MRC BNDU Data Sharing Platform under its own terms; **neither the recordings nor any
per-patient derived values are redistributed here**. `tools/collect_results.py` excludes all Oxford-derived
files by default, and `tools/strip_restricted_outputs.py` removes Oxford cell outputs from an executed
notebook before it is committed.

## Citation

If you use this code, please cite it (see [`CITATION.cff`](CITATION.cff)) and the datasets:

* Rassoulou F, Steina A, Hartmann CJ, Vesper J, Butz M, Schnitzler A, Hirschmann J (2024). Exploring the
  electrophysiology of Parkinson's disease with magnetoencephalography and deep brain recordings.
  *Scientific Data* 11, 889. https://doi.org/10.1038/s41597-024-03768-1 (data: OpenNeuro ds004998, CC0).
* Oxford MRC Brain Network Dynamics Unit, STN LFP ON/OFF and DBS dataset (MRC BNDU Data Sharing Platform);
  see also Wiest C, et al. (2023), *eLife*, on the aperiodic exponent of subthalamic field potentials.

## License

Code: MIT (see `LICENSE`). Results tables and figures: CC BY 4.0.

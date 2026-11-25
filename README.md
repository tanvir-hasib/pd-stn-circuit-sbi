# Levodopa and the STN–GPe circuit: simulation-based inference on STN-LFP and MEG in Parkinson's disease

A cortex–STN–GPe circuit model with an STN high-frequency-oscillation (HFO) generator, fitted to resting
subthalamic local field potentials (STN-LFP) and MEG recorded OFF and ON levodopa, using amortised
simulation-based inference (neural posterior estimation, amortised model comparison and a joint OFF/ON
posterior network). The question: **which circuit parameters does levodopa change, and do they explain the
medication effects on STN β, HFO and β–HFO phase–amplitude coupling (PAC)?**

Authors: Tanvir Hasib¹, Parameswari Shunmugam², Vijayakumar Vengadasalam¹ (¹ Multimedia University, Malaysia;
² Sultan Salahuddin Abdul Aziz Shah Polytechnic, Malaysia). Analysis version **v5.1** (frozen); revision analyses
**v5.1-rev1**, run on the frozen v5.1 networks. Model history in [`CHANGELOG.md`](CHANGELOG.md).

Code and results for the manuscript *Calibrated simulation-based inference links levodopa's suppression of
subthalamic beta oscillations to reduced STN–GPe loop drive in Parkinson's disease* (Hasib, Shunmugam and
Vengadasalam; submitted).

## Main results (v5.1 + revision analyses)

ds004998 (Rassoulou et al., 2024): 37 OFF/ON hemisphere pairs from 19 patients, 36 within the reach of the joint
simulations; Oxford MRC BNDU: 11 OFF/ON pairs (pre-stimulation segments). The model was developed on ds004998, so
the ds004998 inference is model-development-informed, not confirmatory. All mechanistic statements are relative to
the implemented mechanisms within this model class.

| Finding | Evidence | Where |
|---|---|---|
| Levodopa reduces STN β, with patients as the unit | patient-level p < 0.001; mixed-model change −0.075 (95 % CI −0.120 to −0.030) | §8; R1 |
| Validation across the prior overstates identifiability | w_GS decrease AUC 0.91 across the prior, but 0.56–0.71 against noise or slow-input decreases in the patients' parameter region; the network is biased towards reporting w_GS decreases there | §10.6; R5, R5b |
| The cohort shift exceeds session nuisance | sparse-prior cohort mean ×0.864 vs ×0.960 [0.936, 0.982] for nuisance-only cohorts (p < 0.001); 0 of 72 no-change cohorts reached the hierarchical estimate | R5b, R13 B–C |
| Reduced slow burst input is disfavoured | calibrated p = 0.002; best-fitting scenario in ≤ 5 % of patient bootstraps under every set of fixed constants | R5b, R12b, R13 D |
| Noise reduction alone does not reproduce the size of the shift (assumed constants) | 0 of 40 noise-only cohorts reached the data's estimate (hierarchical ×0.85–0.88 vs ×0.75); not tested under misspecified constants | R14 |
| Size of the population change of effective GPe→STN coupling | **×0.76, 90 % CI 0.71–0.82** (bias-corrected on 184 simulated cohorts; frequentist Neyman interval, not a posterior interval); alternative calibrations (concurrent noise reduction, w_GG ×0.5, BG delays ×1.25) give point estimates ×0.76–0.82. Uncorrected: hierarchical ×0.75 (0.68–0.83), sparse-prior cohort mean ×0.86 (0.80–0.93) | R13 C, R14 |
| Posterior signatures favour scenarios with a w_GS decrease, but do not exclude noise | scenarios containing a w_GS decrease are the most probable in 61–75 % of bootstraps under the assumed constants (noise alone 23–24 %) and 49–69 % under misspecified constants (noise alone 15–40 %) | R12b, R13 D |
| The model is structurally incomplete | no implemented scenario reproduces the complete posterior signature (typicality ≤ 0.02; ≤ 0.11 under misspecification) | R12b, R13 D |
| The inferred change reproduces the observed β changes | posterior predictive 90 % coverage of ΔSTN β 0.83 vs 0.50 with the candidate changes ablated | §10.7; R6 |
| The HFO shift is robust but not explained by the implemented β-state gate | fast/slow ratio ↑ (patient-level p = 0.003) under all notch, HPI and artefact exclusions; the gate parameters are not identifiable, so the data say nothing about the strength of any such gate | §10.8; R2 |
| β–HFO PAC is not an independent marker | weak, medication-independent, tracks β and HFO power; PAC–clinical correlations are exploratory | §8b; R3, R9 |
| Oxford: directional consistency only | STN-only joint network ×0.91 (0.79–1.02), uncalibrated, 10 hemispheres | §11; R7 |

Limitations stated in the manuscript: per-hemisphere attribution is not possible in the patients' region; the
w_GS / noise split is partly degenerate; only four fixed-constant misspecifications were tested, and no structural
alternatives (separate GPe populations, striatal or thalamic loops); κ_H and ν_H are not identifiable; w_GS does
not track clinical scores and is not an individual biomarker.

## Repository layout

```
notebooks/pd_stn_circuit_sbi.ipynb            the full v5.1 analysis (Jupyter)
notebooks/pd_stn_circuit_sbi.py               the same, as a percent-format script (# %% cells)
notebooks/pd_stn_circuit_sbi_executed.ipynb   the v5.1 full run with outputs (Oxford cell outputs removed)
revision/                                     revision analyses R1–R14 on the frozen v5.1 networks (see revision/README.md)
revision/executed/                            the reported revision runs with outputs (restricted outputs removed)
data/README.md                                how to obtain the two datasets
results/                                      tables and figures from the full run (see results/README.md)
results/revision/                             tables and figures from the revision analyses
tools/collect_results.py                      copies the shareable outputs from OUT_DIR into results/ (--revision for R1–R14)
tools/strip_restricted_outputs.py             removes Oxford-derived outputs and local paths from an executed notebook
CHANGELOG.md                                  model and analysis history (v4 → v5.1-rev1)
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
5. **Revision analyses (optional, ≈ 12–15 h in total):** run the notebooks in `revision/` in order, with the same
   environment variables and `OUT_DIR` (see [`revision/README.md`](revision/README.md)). They reuse the v5.1 caches
   and retrain nothing.
6. **Collect results:** `python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results --revision`.

## Reproducibility

* Fixed seeds throughout (`SEED = 20261113`); simulation banks, networks and classifiers are cached under a
  hash of the model configuration, prior, statistics and integrator.
* §9 writes a pipeline lock (`pipeline_lock_v51.json`): a hash of the configuration and the source of the
  preprocessing, statistics and simulator functions, taken before the model is fitted to data.
* The CUDA-graph simulator is checked against eager execution at start-up (§2).
* The revision analyses execute the frozen pipeline with its caches and add simulations only for calibration;
  no network, classifier or prior was changed after the lock (`revision/README.md`).
* Exact package versions of the reported run: see `requirements.txt`.

## Data and privacy

No patient data are included. ds004998 is public on OpenNeuro. The Oxford MRC BNDU recordings are available
on request from the MRC BNDU Data Sharing Platform under its own terms; **neither the recordings nor any
per-patient derived values are redistributed here**. `tools/collect_results.py` excludes all Oxford-derived
files by default, and `tools/strip_restricted_outputs.py` removes Oxford cell outputs from an executed
notebook before it is committed (`--scrub-paths` also removes local user paths; `--check-only` scans notebooks
for remaining Oxford references). The revision analysis R7 (Oxford, bipolar channels only) is printed only; its
outputs are removed from `revision/executed/`. The one remaining "Oxford" line in
`revision/executed/revision_analyses_executed.ipynb` (R2) is the gain of the resampling filter for white noise,
not a data value.

## Citation

If you use this code, please cite the archived version (Zenodo DOI in [`CITATION.cff`](CITATION.cff)), the
manuscript once published, and the datasets:

* Rassoulou F, Steina A, Hartmann CJ, Vesper J, Butz M, Schnitzler A, Hirschmann J (2024). Exploring the
  electrophysiology of Parkinson's disease with magnetoencephalography and deep brain recordings.
  *Scientific Data* 11, 889. https://doi.org/10.1038/s41597-024-03768-1 (data: OpenNeuro ds004998, CC0).
* Oxford MRC Brain Network Dynamics Unit, STN LFP ON/OFF and DBS dataset (MRC BNDU Data Sharing Platform);
  see also Wiest C, et al. (2023), *eLife*, on the aperiodic exponent of subthalamic field potentials.

## License

Code: MIT (see `LICENSE`). Results tables and figures: CC BY 4.0.

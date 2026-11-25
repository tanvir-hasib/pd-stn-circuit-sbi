# Results (v5.1 full run and revision analyses)

Populate with `python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results`.
All files are ds004998-derived (public data) or simulation-derived; Oxford-derived files are excluded.

| File | Content (notebook section) |
|---|---|
| `parameter_recovery.csv/.png`, `posterior_correlation.png` | NPE recovery, calibration and identifiability on held-out simulations (§5) |
| `model_recovery.png` | confusion matrix of the 12-hypothesis classifier (§6) |
| `ds_off_on.csv/.png` | ds004998 OFF vs ON statistics (§8) |
| `coupling_full_rest.csv/.png` | calibrated STN β→HFO coupling, both nulls (§8b; only if `RUN_COUPLING`) |
| `ds_fitted_parameters.csv`, `ds_ppc.png` | single-session posteriors and posterior predictive check (§10) |
| `ds_paired_parameter_change.csv`, `ds_paired_parameter_summary.csv`, `ds_param_change.csv` | independent OFF/ON fits, paired parameter change (§10) |
| `ds_model_probabilities.csv/.png` | per-hemisphere hypothesis probabilities, RFX-BMS (§10) |
| `joint_recovery.csv/.png`, `joint_attribution.png`, `joint_attribution_P{dec,inc}.csv` | joint OFF/ON network: held-out recovery and attribution (§10.6) |
| `ds_joint_posterior.csv`, `ds_joint_group.csv`, `ds_joint_forest.png`, `ds_joint_ppc.csv` | joint posterior per hemisphere and cohort (§10.6) |
| `ds_fulljoint_group.csv`, `ds_fulljoint_ppc.csv` | full joint network: cohort changes, PPC with ablation (§10.7) |
| `beta_reach_drivers.csv`, `beta_reach_changes.csv`, `beta_reach_signature.csv`, `beta_reachability.png` | β reachability (§10.7) |
| `ds_gate_ablation.csv`, `gate_ablation.png` | β-state HFO-gate ablation (§10.8) |
| `ds004998_STN-only_joint_{posterior,group}.csv` | STN-only joint network on ds004998, for comparison with Oxford (§11.5b) |
| `ds004998_statistics.csv`, `ds004998_channel_selection.json` | summary statistics and STN contact selection per recording |
| `pipeline_lock_v51.json` | pipeline hash written before fitting (§9) |

## Revision analyses (`results/revision/`)

Populate with `python tools/collect_results.py --out-dir pd_pac_v4_outputs --dest results --revision`. Caches
(`R12_pairs_*.npz`, `R12b_combined_*.npz`, `R13/*.npz`) and Oxford-derived files are not copied.

| File | Content |
|---|---|
| `R1_offon_patient_level.csv`, `R1_hierarchical_changes.csv`, `figures/R1_hierarchical_wGS.png` | patient-level OFF/ON tests; hierarchical changes (R1) |
| `R2_hfo_robustness.csv`, `R2_hfo_variants_per_recording.csv`, `R2_resampling_gain.csv`, `figures/R2_hfo_notch_spectra.png` | HFO notch/HPI/artefact sensitivity (R2) |
| `R3_pac_partial.csv` | PAC partial correlations (R3) |
| `R4_prior_sparsity.csv` | change-prior sparsity (R4) |
| `R5_misspecification.csv`, `figures/R5_misspecification.png`, `R5b_null_calibration.csv` | misspecified constants; calibrated null (R5, R5b) |
| `R6_ppc_sparsity.csv`, `R8_beta_best_pair_metrics.csv`, `R9_clinical.csv` | R6, R8, R9 |
| `R10_*.csv`, `figures/R10_hierarchical_recovery.png`, `R11_hierarchical_sensitivity.csv` | first hierarchical calibration and sensitivity (R10–R11) |
| `R12_*.csv`, `R12b_*.csv`, `figures/R12_scenario_signatures.png`, `figures/R12b_scenarios.png` | posterior-signature comparison (R12, R12b) |
| `R13/A_hemisphere_flow.csv`, `R13/B_*.csv` | hemisphere flow; 36-hemisphere results (R13 A–B) |
| `R13/C_*.csv`, `figures/R13_hierarchical_calibration.png` | calibration of the hierarchical estimator, 184 cohorts (R13 C) |
| `R13/D_signature_stability.csv` | signature stability, 12 pairs (R13 D) |
| `R14/*.csv`, `figures/R14_calibration_alternatives.png` | calibration under alternative true effects (R14) |

# Changelog

Each model revision was triggered by a failed check in the previous full run; the analysis was frozen at v5.1.
Caches are keyed by a configuration hash, so revisions that change the model rebuild every simulation bank and
network.

## v5.1 — β-state HFO-gate ablation (frozen)
* §10.8: the §10.7 posterior draws re-simulated with the fitted HFO gate, a frozen gate (both sessions at the OFF
  β state) and no gate (κ_H = ν_H = 0), with identical circuit runs and noise.
* Result: the gate does not explain the HFO increase ON (gated vs frozen p = .09; κ_H, ν_H posteriors equal the
  prior). The HFO increase is reported as an effect dissociable from the β change. β results are unchanged by the gate.

## v5 — model revision
* **Absolute-scale observation model.** v4 z-scored the circuit field before adding the background, so a circuit
  change that weakened β was normalised away and β power above 1/f could only change through measurement terms
  (shown by the v4.7 reachability analysis). Circuit fields are now divided by fixed reference amplitudes
  (U_REF_S = 50, U_REF_E = 24, B_REF = 50; calibrated at w_GS = 3), and background, HFO and the low-frequency
  component have absolute sizes; SNR is the subject's gain of circuit vs background.
* **β-state gated HFO**: amplitude a_H·E_β^(−κ_H)·(1 + λ_H tanh z(S)), frequency f_H − ν_H ln E_β, with E_β the
  0.5-s smoothed β envelope of the STN input (new parameters κ_H, ν_H). Prior on log₁₀a_H widened to −3.5…0.5.
* All banks, pairs, joint simulations and networks rebuilt.

## v4.7 — full joint network and β reachability
* §10.7: joint posterior over all OFF parameters + 11 candidate changes + 8 session-nuisance changes, giving a
  self-consistent posterior predictive check (OFF, ON, OFF→ON, ablation of the candidate changes).
* β reachability near each hemisphere's OFF state: how often the observed β drop is reached, which changes
  produce it, and whether the accompanying changes match the data (signature test).

## v4.6 — joint OFF/ON network
* §10.6: one posterior network sees both sessions of a hemisphere, x = [S_OFF, S_ON − S_OFF], and returns θ_OFF
  and Δθ for 11 candidate parameters under a neutral, sparse, two-sided change prior. Validated on held-out pairs,
  by attribution on the hypothesis pairs, and by a posterior predictive check. STN-only version for Oxford.

## v4.5.1–v4.5.4 — Oxford loader
* MATLAB v7.3 (HDF5) and v5 reader; channel-title field chosen by channel count; only raw bipolar STN channels of
  the file's side (fallback: a label shared by OFF and ON); flat channels dropped.
* DBS/stimulation detection (sustained narrow line ≥ 15 dB in 110–270 Hz); segments only **before the first
  stimulation block**, searched within the first 300 s of every session; ERNA files excluded.

## v4.5 — factorial hypotheses and Oxford replication
* §6: β route {none, GPe→STN, slow burst input, noise} × HFO shift {no, yes} + hyperdirect, BG→cortex, cortical
  gain, β-locking (12 hypotheses); family-level RFX-BMS and family recovery. (v4.4's winner had been the only
  hypothesis combining a β reduction with an HFO shift.)
* §11: Oxford MRC BNDU STN-only replication with the model locked; DBS described (exploratory).

## v4.4
* Free-bandwidth low-frequency component (3–12 Hz) at both sites; session nuisance in every simulated OFF/ON pair;
  paired posterior test of OFF→ON parameter changes.

## v4.3
* Wider noise floors and SNR range; shared 3–7 Hz tremor component; rejection sampling of in-prior posterior
  samples; boundary diagnostic; leave-one-subject-out and multi-seed robustness of the model comparison.

## v4.2
* STN HFO generator (f_H, a_H, β-locking λ_H); slow GPe→STN gain fluctuation for β bursts (σ_slow); k_d scales BG
  delays and time constants (β 14–35 Hz); HFO statistics; gentle EMG mask; calibrated coupling statistics on the
  full rest period (circular-shift and cross-session nulls).

## v4.1
* Event alignment fix (first-sample offset); identical notches in data and simulations; robust β-envelope
  variability; free BG delay scale; cortical E–I gain; free noise floors; parallel statistics.

## v4 — circuit model on ds004998
* GPU rate model of cortex, STN, GPe and BG→cortex output with Erlang delays (stochastic Heun, 0.2 ms, CUDA graphs);
  LFP/MEG observation model; NPE; amortised model comparison; ds004998 loading and preprocessing.

## Before v4
* v1–v3 (scalp EEG, UCSD dataset): δ–β PAC absent; the β–γ medication effect disappeared after ICA (muscle),
  motivating the move to invasive STN recordings.

# %% [markdown]
# # Revision analyses, part 2: hierarchical-estimate validation and multivariate scenario comparison
#
# Second-round review, frozen v5.1 networks (nothing is retrained):
#
# | § | Review point | Analysis |
# |---|---|---|
# | R10 | 4.1, 2.1 | **Recovery of the hierarchical population estimate** on simulated patient cohorts with the data's structure (19 patients, 36 hemispheres, OFF states from the fitted posteriors) and known population changes of w_GS; bias, 90% coverage and false-positive rate of the hierarchical estimator and of the sparse-prior cohort mean |
# | R11 | 2.1, 4.1 | **Sensitivity of the data's hierarchical estimate**: Student-t and contaminated-normal patient effects, weight truncation, no reweighting, inflated hemisphere variances, Monte Carlo bootstrap of the posterior samples, leave-one-patient-out |
# | R12 | 3, 5 | **Comparison of complete 11-parameter posterior-change signatures** across four scenarios (nuisance only, noise ↓, slow input ↓, w_GS ↓, each over a wide range of magnitudes), under the assumed constants and three misspecified ones; held-out classification accuracy, the data's scenario probabilities with patient-bootstrap intervals, typicality of the data under each scenario, and a figure |
#
# The notebook runs `revision_analyses.py` with all of its analyses switched off, which executes the frozen
# pipeline (with caches) and defines the helper functions used below. Outputs: `OUT_DIR/revision/R10_*.csv`, …,
# figures in `OUT_DIR/revision/figures/`. Runtime ≈ pipeline load + 1.5–2 h (R10 ≈ 30 min, R11 ≈ 10 min, R12 ≈ 60 min).

# %% [markdown]
# ## 0. Settings

# %%
import os
from pathlib import Path

REV_SCRIPT = Path("revision_analyses.py")    # ← round-1 script, same folder
PIPELINE = os.environ.get("PD_PAC_PIPELINE", "../notebooks/pd_stn_circuit_sbi.py")   # ← the frozen v5.1 pipeline
DS_ROOT = os.environ.get("DS004998_ROOT", "../data/ds004998")    # ← as in the v5.1 run
OUT_DIR = os.environ.get("PD_PAC_OUT", "../notebooks/pd_pac_v4_outputs")   # ← SAME output folder as the v5.1 run (caches)
GPU_ID = int(os.environ.get("PD_PAC_GPU", "0"))   # CUDA device index
QUICK = False

N_COHORTS = 8            # R10: simulated cohorts per true population change
TAU_TRUE = 0.09          # R10: between-patient s.d. of the true change (log scale; data estimate 0.09)
SIGMA_TRUE = 0.05        # R10: within-patient (hemisphere) s.d. of the true change
K_PAIRS = 6              # R12: simulated pairs per hemisphere, scenario and condition (half for training, half for testing)
N_BOOT = 1000            # R11/R12: bootstrap replicates
SEED2 = 20261121
R12_CONDITIONS = ["assumed (no misspecification)", "w_GG ×0.5", "τ_S, τ_G ×0.8", "BG delays ×1.25"]

# %% [markdown]
# ## 1. Load the frozen pipeline and the round-1 helper functions

# %%
import re

src = REV_SCRIPT.read_text(encoding="utf-8")
OVR = {"PIPELINE": f"Path(r\"{PIPELINE}\")", "DS_ROOT": f"r\"{DS_ROOT}\"", "OUT_DIR": f"r\"{OUT_DIR}\"", "GPU_ID": str(GPU_ID),
       "QUICK": str(QUICK), "RUN_OXFORD_REV": "False", "RUN_ONLY": "set()"}
for name, value in OVR.items():
    line = f"{name} = {value}"
    src, n = re.subn(rf"^{name}\s*=.*$", lambda _m, line=line: line, src, count=1, flags=re.M)
    assert n == 1, f"could not find the setting {name} in {REV_SCRIPT}"
exec(compile(src, str(REV_SCRIPT), "exec"), globals())
RUN_ONLY = None                                       # re-enable run() for the analyses below (or e.g. {"R12"})

# %%
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf

KEYS = [k for k in FS_DS if k in JS_DS]               # hemispheres with both joint posteriors (36)
SUBJ = np.array([k[0] for k in KEYS])
KG = JP.index("w_GS"); G0 = len(JP); WG = J_W[KG]
print(f"{len(KEYS)} hemispheres from {len(set(SUBJ))} patients")


def sample_pairs(TO, TN, seed, n_samp=2000):
    """Simulate OFF/ON pairs and return the joint-network posterior samples of the 11 changes for each pair."""
    S = simulate_stats(np.concatenate([TO, TN]), seed)
    F = pair_features(S[:len(TO)], S[len(TO):]); ok = np.isfinite(F).all(1)
    D = [joint_samples(JPOST, F[i], n_samp)[:, G0:] if ok[i] else None for i in range(len(F))]
    return D, ok


def off_draws(rng, k):
    s_ = FS_DS[k]
    return np.clip(s_[rng.integers(0, len(s_), 1), :nP], PRIOR_LO, PRIOR_HI)


def cohort_mean_factor(Ds, n=4000, seed=0):
    """Sparse-prior cohort-mean estimator used in the paper (one draw per hemisphere, averaged)."""
    r = np.random.default_rng(seed)
    G = np.mean([d[r.integers(0, len(d), n)] for d in Ds], 0)
    return np.median(G), np.percentile(G, 5), np.percentile(G, 95), (G < 0).mean()

# %% [markdown]
# ## R10. Does the hierarchical model recover known population effects?
#
# For each simulated cohort, every hemisphere of the data receives an OFF state drawn from its own posterior and a
# true w_GS change Δ_ij = μ + u_i + ε_ij (u_i: patient effect, s.d. TAU_TRUE; ε_ij: hemisphere effect, s.d.
# SIGMA_TRUE); session nuisance is drawn as in training and no other candidate changes. The pipeline is then applied
# exactly as to the data. μ ∈ {0, log 0.9, log 0.8, log 0.7}, plus μ = log 0.8 with heavy-tailed (Student-t, 3 d.f.)
# patient effects.

# %%
R10_DESIGN = [("μ = ×1.00", 0.0, "normal"), ("μ = ×0.90", np.log(.9), "normal"), ("μ = ×0.80", np.log(.8), "normal"),
              ("μ = ×0.70", np.log(.7), "normal"), ("μ = ×0.80, t3 patient effects", np.log(.8), "t3")]


def R10():
    rng = np.random.default_rng(SEED2); subs = pd.unique(SUBJ)
    TO, TN, meta = [], [], []
    for lab, mu, dist in R10_DESIGN:
        for c in range(N_COHORTS):
            u = dict(zip(subs, TAU_TRUE * (rng.standard_t(3, len(subs)) / np.sqrt(3) if dist == "t3" else rng.standard_normal(len(subs)))))
            for h, k in enumerate(KEYS):
                d = float(np.clip(mu + u[k[0]] + SIGMA_TRUE * rng.standard_normal(), -0.95 * WG, 0.95 * WG))
                th = off_draws(rng, k); Dm = np.zeros((1, nM)); Dm[0, KG] = d
                TO.append(th); TN.append(apply_full(th, Dm, nuisance_draw(th, rng))); meta.append((lab, c, h, d))
    TO, TN = np.concatenate(TO), np.concatenate(TN)
    print(f"simulating {len(TO)} pairs ({2 * len(TO)} simulations)")
    D, ok = sample_pairs(TO, TN, SEED2 + 10)
    rows = []
    for lab, mu, dist in R10_DESIGN:
        for c in range(N_COHORTS):
            idx = [i for i, m in enumerate(meta) if m[0] == lab and m[1] == c and ok[i]]
            Ds = [D[i][:, KG] for i in idx]; true_d = np.array([meta[i][3] for i in idx])
            sp = cohort_mean_factor(Ds, seed=c)
            mv = np.array([flat_likelihood(d, WG) for d in Ds])
            h, _ = hierarchical_normal(mv[:, 0], mv[:, 1], SUBJ[[meta[i][2] for i in idx]], WG)
            rows.append(dict(design=lab, cohort=c, true_mu=mu, true_sample_mean=true_d.mean(),
                             sparse_est=sp[0], sparse_lo=sp[1], sparse_hi=sp[2], sparse_P=sp[3],
                             hier_est=h["mu_med"], hier_lo=h["mu_lo"], hier_hi=h["mu_hi"], hier_P=h["P_mu_neg"], hier_tau=h["tau_med"]))
    T = pd.DataFrame(rows); T.to_csv(REV / "R10_hierarchical_recovery_cohorts.csv", index=False)
    S_ = T.groupby("design", sort=False).apply(lambda g: pd.Series(dict(
        true=np.exp(g.true_mu.iloc[0]),
        sparse_mean=np.exp(g.sparse_est.mean()), sparse_bias_log=(g.sparse_est - g.true_sample_mean).mean(),
        sparse_cov90=np.mean((g.sparse_lo <= g.true_sample_mean) & (g.true_sample_mean <= g.sparse_hi)),
        sparse_P_gt_0_95=np.mean(g.sparse_P > .95),
        hier_mean=np.exp(g.hier_est.mean()), hier_bias_log=(g.hier_est - g.true_mu).mean(), hier_rmse_log=np.sqrt(((g.hier_est - g.true_mu) ** 2).mean()),
        hier_cov90=np.mean((g.hier_lo <= g.true_mu) & (g.true_mu <= g.hier_hi)),
        hier_P_gt_0_95=np.mean(g.hier_P > .95), hier_tau_mean=g.hier_tau.mean())))
    print("Recovery on simulated cohorts (data structure; log-scale bias; cov90 = 90% interval contains the truth;\n"
          "P_gt_0_95 = fraction of cohorts with P(decrease) > 0.95, i.e. the false-positive rate when the truth is ×1.00)")
    print(S_.round(3).to_string()); S_.to_csv(REV / "R10_hierarchical_recovery_summary.csv"); SUMMARY["R10"] = S_
    # calibration line: estimate vs truth
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for est, lo, hi, col, lab in (("sparse_est", "sparse_lo", "sparse_hi", "#8c8c8c", "sparse-prior cohort mean"),
                                  ("hier_est", "hier_lo", "hier_hi", "#2a78d6", "hierarchical")):
        x = np.exp(T.true_mu) * (0.99 if col == "#8c8c8c" else 1.01)
        ax.errorbar(x, np.exp(T[est]), yerr=[np.exp(T[est]) - np.exp(T[lo]), np.exp(T[hi]) - np.exp(T[est])],
                    fmt="o", ms=3, color=col, alpha=.6, lw=.6, label=lab)
    ax.plot([.65, 1.05], [.65, 1.05], "k--", lw=.6)
    ax.set_xlabel(r"true population $w_{GS}$ ON/OFF"); ax.set_ylabel("estimate (90% interval)")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(REVF / "R10_hierarchical_recovery.png", dpi=300); plt.show()


run(R10)

# %% [markdown]
# ## R11. Sensitivity of the data's hierarchical estimate
#
# The hierarchical model is refitted by numerical integration over the patient effect, which allows non-normal
# patient effects: Student-t (3 d.f.) and a contaminated normal (80% N(0, τ²) + 20% N(0, (3τ)²)). The normal case is
# fitted the same way as a check against the closed form. Reweighting variants: weights truncated at mean·√N
# (Ionides, 2008); no reweighting (moments of the sparse-prior posterior, which carry its shrinkage); hemisphere
# variances doubled; Monte Carlo bootstrap of the posterior samples; leave-one-patient-out.

# %%
def hier_grid(m, v, subj, w, re="normal", n_mu=241, n_s=30, n_z=121):
    """Hierarchical model with patient effect u_i = τ z_i, z_i ~ normal / t3 / contaminated normal; grid posterior of μ."""
    mu_g = np.linspace(-w, w, n_mu); s_g = np.linspace(1e-3, w, n_s)
    L = 6.0 if re == "normal" else 15.0
    z = np.linspace(-L, L, n_z); dz = z[1] - z[0]
    if re == "normal":
        lpz = stats.norm.logpdf(z)
    elif re == "t3":
        lpz = stats.t.logpdf(z * np.sqrt(3.0), 3) + 0.5 * np.log(3.0)   # t3 scaled to unit variance: τ is the s.d.
    else:
        lpz = np.logaddexp(np.log(.8) + stats.norm.logpdf(z), np.log(.2) + stats.norm.logpdf(z, scale=3.0))
    lpz = lpz + np.log(dz)
    subs = pd.unique(subj); prior_s = -0.5 * (s_g / (w / 2)) ** 2
    logp = np.zeros((n_mu, n_s, n_s))
    for si, sg in enumerate(s_g):
        for ti, tg in enumerate(s_g):
            ll = np.zeros(n_mu)
            for s_ in subs:
                k = subj == s_; var = sg ** 2 + v[k]                               # hemispheres of patient s_
                r = m[k][None, None, :] - mu_g[:, None, None] - tg * z[None, :, None]   # (mu, z, hemi)
                lj = (-0.5 * (r ** 2 / var + np.log(2 * np.pi * var))).sum(-1)       # (mu, z)
                ll += logsumexp(lj + lpz[None, :], axis=1)
            logp[:, ti, si] = ll + prior_s[si] + prior_s[ti]
    P = np.exp(logp - logp.max()); P /= P.sum(); pm = P.sum((1, 2)); cdf = np.cumsum(pm)
    q = lambda a: mu_g[min(np.searchsorted(cdf, a), n_mu - 1)]
    return dict(mu_med=q(.5), mu_lo=q(.05), mu_hi=q(.95), P_mu_neg=pm[mu_g < 0].sum())


def flat_moments(d, w, mode="flat"):
    if mode == "none":
        return d.mean(), d.var(), float(len(d))
    wt = 1.0 / np.maximum(p0_density(d, w), 1e-300)
    if mode == "truncated":
        wt = np.minimum(wt, wt.mean() * np.sqrt(len(wt)))
    wt = wt / wt.sum(); mm = np.sum(wt * d)
    return mm, np.sum(wt * (d - mm) ** 2), 1 / np.sum(wt ** 2)


def R11():
    rows = []
    for net, SS, col in (("STN+MEG network", JS_DS, G0 + KG), ("full network", FS_DS, nP + KG)):
        keys = list(SS); subj = np.array([k[0] for k in keys])
        Ds = [SS[k][:, col] for k in keys]
        mv = {mode: np.array([flat_moments(d, WG, mode) for d in Ds]) for mode in ("flat", "truncated", "none")}

        def add(variant, h):
            rows.append(dict(network=net, variant=variant, factor=np.exp(h["mu_med"]), lo90=np.exp(h["mu_lo"]),
                             hi90=np.exp(h["mu_hi"]), P_decrease=h["P_mu_neg"]))
        h0, _ = hierarchical_normal(mv["flat"][:, 0], mv["flat"][:, 1], subj, WG); add("normal patient effects (paper, closed form)", h0)
        for re_, lab in (("normal", "normal patient effects (numerical, check)"), ("t3", "Student-t (3 d.f.) patient effects"),
                         ("contam", "contaminated normal patient effects (20% × 3τ)")):
            add(lab, hier_grid(mv["flat"][:, 0], mv["flat"][:, 1], subj, WG, re_))
        add("weights truncated at mean·√N", hierarchical_normal(mv["truncated"][:, 0], mv["truncated"][:, 1], subj, WG)[0])
        add("no reweighting (sparse-prior moments)", hierarchical_normal(mv["none"][:, 0], mv["none"][:, 1], subj, WG)[0])
        add("hemisphere variances ×2", hierarchical_normal(mv["flat"][:, 0], 2 * mv["flat"][:, 1], subj, WG)[0])
        # Monte Carlo bootstrap of the posterior samples
        r = np.random.default_rng(SEED2 + 11); bs = []
        for _ in range(40):
            mvb = np.array([flat_moments(d[r.integers(0, len(d), len(d))], WG) for d in Ds])
            bs.append(hierarchical_normal(mvb[:, 0], mvb[:, 1], subj, WG)[0]["mu_med"])
        rows.append(dict(network=net, variant="Monte Carlo bootstrap of posterior samples (40×): 5–95% of estimate",
                         factor=np.exp(np.median(bs)), lo90=np.exp(np.percentile(bs, 5)), hi90=np.exp(np.percentile(bs, 95)), P_decrease=np.nan))
        lo = []
        for s_ in pd.unique(subj):
            k = subj != s_
            lo.append(hierarchical_normal(mv["flat"][k, 0], mv["flat"][k, 1], subj[k], WG)[0])
        rows.append(dict(network=net, variant="leave-one-patient-out: range of estimate (and of P)",
                         factor=np.exp(np.median([h["mu_med"] for h in lo])), lo90=np.exp(min(h["mu_med"] for h in lo)),
                         hi90=np.exp(max(h["mu_med"] for h in lo)), P_decrease=min(h["P_mu_neg"] for h in lo)))
        print(f"  {net}: min ESS (flat) {mv['flat'][:, 2].min():.0f}, (truncated) {mv['truncated'][:, 2].min():.0f} of {len(Ds[0])}")
    T = pd.DataFrame(rows); print("Population w_GS ON/OFF factor of the data under alternative hierarchical assumptions:")
    print(T.round(3).to_string(index=False)); T.to_csv(REV / "R11_hierarchical_sensitivity.csv", index=False); SUMMARY["R11"] = T


run(R11)

# %% [markdown]
# ## R12. Which scenario does the data's complete posterior-change signature match?
#
# For each condition (assumed constants, and three misspecifications), K_PAIRS pairs per hemisphere and scenario are
# simulated with OFF states from the patients' posteriors; magnitudes cover a wide range so that the comparison does
# not depend on one effect size:
# * nuisance only;  noise ↓: Δlog₁₀σ = −U(0.05, 0.25);  slow input ↓: σ_slow × U(0.3, 0.85);  w_GS ↓: × U(0.6, 0.95).
#
# The **signature** of a cohort is the vector of mean posterior changes of all 11 candidates (one pair per hemisphere,
# the data's composition). Pairs are split per hemisphere into training and test halves; a shrinkage LDA trained on
# cohorts built from the training halves is tested on cohorts built from the test halves, which gives the held-out
# accuracy and the confusion matrix. Applied to the data, it gives scenario probabilities (equal prior), with 90%
# intervals and the frequency with which each scenario is the most probable from a bootstrap over patients.
# Typicality: fraction of simulated cohorts of each scenario whose Mahalanobis distance from the scenario mean is
# at least the data's (a goodness-of-fit p value; small = the data are atypical of that scenario).
# Robustness to misspecification: the classifier trained under the assumed constants is applied to cohorts
# simulated under each misspecification.

# %%
SCEN = ["nuisance only", "noise ↓", "slow input ↓", "w_GS ↓"]


def scenario_delta(sc, rng):
    Dm = np.zeros((1, nM))
    if sc == "noise ↓":
        Dm[0, JP.index("log10_sig")] = -rng.uniform(.05, .25)
    elif sc == "slow input ↓":
        Dm[0, JP.index("sig_slow")] = np.log(rng.uniform(.3, .85))
    elif sc == "w_GS ↓":
        Dm[0, KG] = np.log(rng.uniform(.6, .95))
    return Dm


def build_cohorts(MU, lab, hk, half, n, rng):
    """n cohort signatures per scenario; pairs restricted to `half` (0 = training, 1 = test)."""
    X, y = [], []
    for si, sc in enumerate(SCEN):
        pools = [np.where((lab == sc) & (hk == h) & half)[0] for h in range(len(KEYS))]
        pools = [p for p in pools if len(p)]
        for _ in range(n):
            X.append(np.mean([MU[p[rng.integers(len(p))]] for p in pools], 0)); y.append(si)
    return np.array(X), np.array(y)


def R12():
    base_c = dict(CIRC); RES = {}
    MU_dat = np.array([JS_DS[k][:, G0:].mean(0) for k in KEYS]); sig_dat = MU_dat.mean(0)
    for ci, cond in enumerate(R12_CONDITIONS):
        rng = np.random.default_rng(SEED2 + 100 * ci)
        TO, TN, lab, hk, half = [], [], [], [], []
        for sc in SCEN:
            for h, k in enumerate(KEYS):
                for r_ in range(K_PAIRS):
                    th = off_draws(rng, k)
                    TO.append(th); TN.append(apply_full(th, scenario_delta(sc, rng), nuisance_draw(th, rng)))
                    lab.append(sc); hk.append(h); half.append(r_ < K_PAIRS // 2)
        TO, TN, lab, hk, half = np.concatenate(TO), np.concatenate(TN), np.array(lab), np.array(hk), np.array(half)
        try:
            for kk, v in MISSPEC[cond].items():
                CIRC[kk] = base_c[kk] * v
            print(f"[{cond}] simulating {len(TO)} pairs")
            D, ok = sample_pairs(TO, TN, SEED2 + 1000 + ci, n_samp=1000)
        finally:
            CIRC.clear(); CIRC.update(base_c)
        MU = np.array([d.mean(0) if d is not None else np.full(len(JP), np.nan) for d in D]); good = ok & np.isfinite(MU).all(1)
        lab, hk, half, MU = lab[good], hk[good], half[good], MU[good]
        RES[cond] = dict(MU=MU, lab=lab, hk=hk, half=half)
        np.savez(REV / f"R12_pairs_{ci}.npz", MU=MU, lab=lab, hk=hk, half=half)

    rows, conf_rows, robust_rows, boot_store = [], [], [], {}
    subs = pd.unique(SUBJ)
    for cond in R12_CONDITIONS:
        R_ = RES[cond]; rng = np.random.default_rng(SEED2 + 7)
        Xtr, ytr = build_cohorts(R_["MU"], R_["lab"], R_["hk"], R_["half"], 600, rng)
        Xte, yte = build_cohorts(R_["MU"], R_["lab"], R_["hk"], ~R_["half"], 300, rng)
        mu_, sd_ = Xtr.mean(0), Xtr.std(0)
        clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit((Xtr - mu_) / sd_, ytr)
        pred = clf.predict((Xte - mu_) / sd_)
        C = pd.crosstab(pd.Series([SCEN[i] for i in yte], name="true"), pd.Series([SCEN[i] for i in pred], name="classified"),
                        normalize="index").reindex(index=SCEN, columns=SCEN, fill_value=0)
        print(f"\n[{cond}] held-out cohort classification accuracy {np.mean(pred == yte):.3f}"); print(C.round(3).to_string())
        conf_rows.append(C.assign(condition=cond))
        # data
        p_dat = clf.predict_proba(((sig_dat - mu_) / sd_)[None])[0]
        rb = np.random.default_rng(SEED2 + 9); PB = []
        for _ in range(N_BOOT):
            ss = rb.choice(subs, len(subs)); idx = np.concatenate([np.where(SUBJ == s_)[0] for s_ in ss])
            PB.append(clf.predict_proba(((MU_dat[idx].mean(0) - mu_) / sd_)[None])[0])
        PB = np.array(PB); boot_store[cond] = PB
        # typicality (Mahalanobis, shrunk covariance of training cohorts of each scenario)
        typ = []
        for si in range(len(SCEN)):
            Xs = (Xtr[ytr == si] - mu_) / sd_; mm = Xs.mean(0)
            ci_ = np.linalg.pinv(LedoitWolf().fit(Xs).covariance_)
            dm = lambda x: float((x - mm) @ ci_ @ (x - mm))
            dd = dm((sig_dat - mu_) / sd_)
            Xt = (Xte[yte == si] - mu_) / sd_
            typ.append(np.mean([dm(x) >= dd for x in Xt]))
        for si, sc in enumerate(SCEN):
            rows.append(dict(condition=cond, scenario=sc, P_data=p_dat[si], P_boot_5=np.percentile(PB[:, si], 5),
                             P_boot_95=np.percentile(PB[:, si], 95), frac_boot_most_probable=np.mean(PB.argmax(1) == si),
                             typicality_p=typ[si], heldout_recall=C.loc[sc, sc]))
        # robustness: classifier trained under the assumed constants, applied to this condition's test cohorts
        if cond != R12_CONDITIONS[0]:
            A = RES[R12_CONDITIONS[0]]; ra = np.random.default_rng(SEED2 + 8)
            Xa, ya = build_cohorts(A["MU"], A["lab"], A["hk"], A["half"], 600, ra); ma, sa = Xa.mean(0), Xa.std(0)
            ca = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit((Xa - ma) / sa, ya)
            pr = ca.predict((Xte - ma) / sa)
            for si, sc in enumerate(SCEN):
                robust_rows.append(dict(condition=cond, true_scenario=sc,
                                        **{f"classified_{s2}": np.mean(pr[yte == si] == j) for j, s2 in enumerate(SCEN)}))
        if cond == R12_CONDITIONS[0]:
            fig_data = (clf, mu_, sd_, Xte, yte, sig_dat, MU_dat)
    T = pd.DataFrame(rows); T.to_csv(REV / "R12_scenario_probabilities.csv", index=False)
    pd.concat(conf_rows).to_csv(REV / "R12_confusion.csv")
    print("\nData: scenario probabilities from the complete 11-parameter signature (equal prior; 90% patient-bootstrap interval),\n"
          "frequency of being the most probable scenario across bootstraps, and typicality p of the data under each scenario")
    print(T.round(3).to_string(index=False)); SUMMARY["R12"] = T
    if robust_rows:
        Rb = pd.DataFrame(robust_rows); Rb.to_csv(REV / "R12_misspecified_with_assumed_classifier.csv", index=False)
        print("\nClassifier trained under the assumed constants applied to cohorts simulated under misspecified constants\n"
              "(rows: true scenario; columns: fraction classified as each scenario):")
        print(Rb.round(3).to_string(index=False)); SUMMARY["R12_robust"] = Rb
    # figure: LDA projection (assumed constants) + scenario probabilities per condition
    clf, mu_, sd_, Xte, yte, sig_dat, MU_dat = fig_data
    Z = clf.transform((Xte - mu_) / sd_)[:, :2]; zd = clf.transform(((sig_dat - mu_) / sd_)[None])[0, :2]
    rb = np.random.default_rng(1); ZB = []
    for _ in range(300):
        ss = rb.choice(subs, len(subs)); idx = np.concatenate([np.where(SUBJ == s_)[0] for s_ in ss])
        ZB.append(clf.transform(((MU_dat[idx].mean(0) - mu_) / sd_)[None])[0, :2])
    ZB = np.array(ZB); cols = ["#8c8c8c", "#eb6834", "#1baf7a", "#2a78d6"]
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for si, sc in enumerate(SCEN):
        axs[0].scatter(Z[yte == si, 0], Z[yte == si, 1], s=4, alpha=.35, color=cols[si], label=sc, lw=0)
    axs[0].scatter(ZB[:, 0], ZB[:, 1], s=3, color="k", alpha=.15, lw=0)
    axs[0].scatter(*zd, s=60, marker="*", color="k", label="data (+ patient bootstrap)", zorder=5)
    axs[0].set_xlabel("discriminant 1"); axs[0].set_ylabel("discriminant 2"); axs[0].legend(frameon=False, fontsize=6, markerscale=2)
    axs[0].set_title("Simulated cohort signatures (assumed constants)", fontsize=8)
    x = np.arange(len(R12_CONDITIONS)); wdt = 0.2
    for si, sc in enumerate(SCEN):
        s = T[T.scenario == sc].set_index("condition").loc[R12_CONDITIONS]
        axs[1].bar(x + (si - 1.5) * wdt, s.P_data, wdt, color=cols[si], label=sc)
        axs[1].errorbar(x + (si - 1.5) * wdt, s.P_data, yerr=[s.P_data - s.P_boot_5, s.P_boot_95 - s.P_data], fmt="none", ecolor="k", lw=.6)
    axs[1].set_xticks(x, [c.replace("assumed (no misspecification)", "assumed") for c in R12_CONDITIONS], fontsize=7, rotation=20, ha="right")
    axs[1].set_ylabel("P(scenario | data signature)"); axs[1].set_ylim(0, 1)
    axs[1].set_title("Data, by assumed fixed constants", fontsize=8)
    fig.tight_layout(); fig.savefig(REVF / "R12_scenario_signatures.png", dpi=300); plt.show()


run(R12)

# %% [markdown]
# ## Summary

# %%
for k, v in SUMMARY.items():
    print(f"\n=== {k} ===")
    print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
print(f"\ntables: {REV}\nfigures: {REVF}")

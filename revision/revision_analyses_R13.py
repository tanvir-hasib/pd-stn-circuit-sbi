# %% [markdown]
# # R13. Third-round review: hemisphere flow, 36-hemisphere consistency, precise hierarchical calibration, signature stability
#
# | § | Review point | Analysis |
# |---|---|---|
# | A | 3 | **Hemisphere flow**: counts at each stage; which hemisphere lies outside the joint simulations' reach, and why |
# | B | 3 | **One hemisphere set (36) throughout**: hierarchical estimates (both networks), sparse-prior cohort mean and the calibrated null test (R5b) recomputed on the 36 in-reach hemispheres |
# | C | 1 | **Precise calibration of the hierarchical estimator**: ≈ 180 simulated cohorts (40 at no change), patient heterogeneity τ = 0.09 and 0.18, normal and heavy-tailed patient effects; bias, 90% coverage and false-positive rate with 95% Monte Carlo intervals; a calibration curve with a Neyman-belt interval for the data's population change; the data's estimate against the null distribution of estimates |
# | D | 2 | **Stability of the signature comparison**: 6 more pairs per hemisphere, scenario and condition (12 in total, 6 per half); scenario ranking and typicality at 6 vs 12 pairs |
#
# Reuses `R12_pairs_*.npz` and `R12b_combined_*.npz` (same OUT_DIR). Every simulation step saves its results and
# resumes from them if the notebook is re-run. Runtime ≈ pipeline load + 4–5 h (C ≈ 3 h, D ≈ 1 h, B ≈ 15 min).

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

# C: (label, population change factor, patient-effect distribution, τ, number of cohorts)
C_DESIGN = [("×1.00", 1.00, "normal", 0.09, 40), ("×0.95", 0.95, "normal", 0.09, 24), ("×0.90", 0.90, "normal", 0.09, 24),
            ("×0.80", 0.80, "normal", 0.09, 16), ("×0.70", 0.70, "normal", 0.09, 16),
            ("×1.00, τ=0.18", 1.00, "normal", 0.18, 16), ("×0.80, τ=0.18", 0.80, "normal", 0.18, 16),
            ("×1.00, t3", 1.00, "t3", 0.09, 16), ("×0.80, t3", 0.80, "t3", 0.09, 16)]
COHORT_SCALE = 1.0       # multiply all cohort counts (e.g. 0.5 for a quicker run)
SIGMA_TRUE = 0.05        # within-patient (hemisphere) s.d. of the true change
K_OLD, K_NEW = 6, 6      # D: pairs per hemisphere already simulated (R12/R12b) and to add
N_BOOT = 1000
SEED4 = 20261123
R12_CONDITIONS = ["assumed (no misspecification)", "w_GG ×0.5", "τ_S, τ_G ×0.8", "BG delays ×1.25"]   # same order as R12
RUN_PARTS = {"A", "B", "C", "D"}

# %% [markdown]
# ## 1. Load the frozen pipeline and helper functions

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
RUN_ONLY = None

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from statsmodels.stats.proportion import proportion_confint

KEYS36 = [k for k in FS_DS if k in JS_DS]               # in-reach hemispheres (full-network order)
SUBJ = np.array([k[0] for k in KEYS36]); SUBS = pd.unique(SUBJ)
KG = JP.index("w_GS"); G0 = len(JP); WG = J_W[KG]
R13 = REV / "R13"; R13.mkdir(exist_ok=True)
print(f"in-reach hemispheres: {len(KEYS36)} from {len(SUBS)} patients; JS_DS has {len(JS_DS)}")


def part(name):
    def deco(fn):
        def wrapped():
            if name in RUN_PARTS:
                fn()
            else:
                print(f"[part {name} skipped]")
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


def off_draw(rng, k, n=1):
    s_ = FS_DS[k]
    return np.clip(s_[rng.integers(0, len(s_), n), :nP], PRIOR_LO, PRIOR_HI)


def posterior_changes(TO, TN, seed, n_samp=1000):
    """Simulate pairs; return per-pair posterior samples of the 11 changes (None if not finite)."""
    S = simulate_stats(np.concatenate([TO, TN]), seed)
    F = pair_features(S[:len(TO)], S[len(TO):]); ok = np.isfinite(F).all(1)
    return [joint_samples(JPOST, F[i], n_samp)[:, G0:] if ok[i] else None for i in range(len(F))]


def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson"); return lo, hi

# %% [markdown]
# ## A. Hemisphere flow

# %%
@part("A")
def A_flow():
    n_sess = len(M_DS); n_hemi = M_DS.groupby(["subject", "side"]).ngroups; n_pat = M_DS.subject.nunique()
    pairs = paired_frame(M_DS, "stn_beta_res")
    rows = [("hemisphere-sessions with a clean 120-s segment", n_sess, n_pat),
            ("hemispheres with a clean segment", n_hemi, n_pat),
            ("hemispheres with both sessions (descriptive OFF/ON analyses)", len(pairs), pairs.subject.nunique())]
    if "FITg" in globals():
        rows.append(("sessions fitted by the single-session posterior (in distribution)", len(FITg), FITg.subject.nunique()))
    rows.append(("pairs with a joint posterior (all paired hemispheres)", len(JS_DS), len({k[0] for k in JS_DS})))
    rows.append(("pairs within the joint simulations' reach (joint inference, PPC, calibration)", len(KEYS36), len(SUBS)))
    T = pd.DataFrame(rows, columns=["stage", "n", "patients"]); print(T.to_string(index=False))
    T.to_csv(R13 / "A_hemisphere_flow.csv", index=False)
    out = [k for k in JS_DS if k not in KEYS36]
    if "JSUM_DS" in globals():
        print("\nout-of-reach pairs (Mahalanobis distance of [OFF statistics, OFF→ON change] beyond the joint simulations' 99th percentile):")
        print(JSUM_DS[JSUM_DS.ood][["subject", "side"]].to_string(index=False))
    for k in out:
        o = M_DS[(M_DS.subject == k[0]) & (M_DS.side == k[1])].set_index("med")
        Fk = pair_features(o.loc[["MedOff"], STAT_NAMES].values.astype(float), o.loc[["MedOn"], STAT_NAMES].values.astype(float))
        Xs = J_F[:J_N]; Xs = Xs[np.isfinite(Xs).all(1)]
        z = (Fk[0] - Xs.mean(0)) / Xs.std(0)
        names = [f"OFF {s}" for s in STAT_NAMES] + [f"Δ {s}" for s in STAT_NAMES]
        top = np.argsort(-np.abs(z))[:6]
        print(f"  {k}: most extreme features (z relative to the joint simulations): " + ", ".join(f"{names[j]} {z[j]:+.1f}" for j in top))
        d = paired_frame(M_DS, "stn_beta_res"); r = d[(d.subject == k[0]) & (d.side == k[1])]
        if len(r):
            print(f"  its STN β: OFF {r.off.iloc[0]:.3f} → ON {r.on.iloc[0]:.3f}")


run(A_flow)

# %% [markdown]
# ## B. One hemisphere set (36) throughout
#
# Recomputes on the 36 in-reach hemispheres: the hierarchical estimate (STN+MEG network: previously 37; full network:
# already 36), the sparse-prior cohort mean (already 36 in the paper), and the calibrated null test (R5b), whose data
# value previously used 37 hemispheres while its simulated cohorts used 36.

# %%
@part("B")
def B_consistency():
    rows = []
    sets = [("STN+MEG network", JS_DS, G0 + KG, f"all paired ({len(JS_DS)})", list(JS_DS)),
            ("STN+MEG network", JS_DS, G0 + KG, f"in reach ({len(KEYS36)})", KEYS36),
            ("full network", FS_DS, nP + KG, f"in reach ({len(KEYS36)})", KEYS36)]
    for net, SS, col, lab, keys in sets:
        if True:
            Ds = [SS[k][:, col] for k in keys]; subj = np.array([k[0] for k in keys])
            mv = np.array([flat_likelihood(d, WG) for d in Ds])
            h, _ = hierarchical_normal(mv[:, 0], mv[:, 1], subj, WG)
            r_ = np.random.default_rng(0); G = np.mean([d[r_.integers(0, len(d), 4000)] for d in Ds], 0)
            rows.append(dict(network=net, hemispheres=lab, n=len(keys), hier=np.exp(h["mu_med"]), hier_lo90=np.exp(h["mu_lo"]),
                             hier_hi90=np.exp(h["mu_hi"]), P_hier=h["P_mu_neg"], tau=h["tau_med"],
                             sparse=np.exp(np.median(G)), sparse_lo90=np.exp(np.percentile(G, 5)), sparse_hi90=np.exp(np.percentile(G, 95)),
                             P_sparse=(G < 0).mean()))
    T = pd.DataFrame(rows); print("w_GS population/cohort change, by hemisphere set:"); print(T.round(3).to_string(index=False))
    T.to_csv(R13 / "B_hemisphere_sets.csv", index=False); SUMMARY["B"] = T

    # calibrated null test (as R5b), data on the same 36 hemispheres as the simulated cohorts
    f = R13 / "B_null_pairs.npz"
    scen = {"nuisance only": 6, "noise ↓": 3, "slow input ↓": 3, "w_GS ↓": 3}
    if f.exists():
        z = np.load(f, allow_pickle=True); lab, hk, MU, DR = z["lab"], z["hk"], z["MU"], list(z["DR"])
        print(f"calibration pairs loaded from {f.name}")
    else:
        rng = np.random.default_rng(SEED4 + 55); TO, TN, lab, hk = [], [], [], []
        for sc, reps in scen.items():
            for h, k in enumerate(KEYS36):
                th = off_draw(rng, k, reps); Dm = np.zeros((reps, nM))
                if sc == "noise ↓":
                    Dm[:, JP.index("log10_sig")] = -rng.uniform(.05, .15, reps)
                elif sc == "slow input ↓":
                    Dm[:, JP.index("sig_slow")] = np.log(rng.uniform(.5, .8, reps))
                elif sc == "w_GS ↓":
                    Dm[:, KG] = np.log(rng.uniform(.7, .9, reps))
                TO.append(th); TN.append(apply_full(th, Dm, nuisance_draw(th, rng))); lab += [sc] * reps; hk += [h] * reps
        TO, TN = np.concatenate(TO), np.concatenate(TN); print(f"simulating {len(TO)} calibration pairs")
        D = posterior_changes(TO, TN, SEED4 + 5500)
        ok = np.array([d is not None for d in D]); lab, hk = np.array(lab)[ok], np.array(hk)[ok]
        D = [d for d in D if d is not None]
        MU = np.array([d.mean(0) for d in D]); DR = [d[:, KG].astype(np.float32) for d in D]
        np.savez(f, lab=lab, hk=hk, MU=MU, DR=np.array(DR, dtype=object))
    r_ = np.random.default_rng(0)
    obs = {n_: np.exp(np.median(np.mean([SS_[k][r_.integers(0, len(SS_[k]), 4000), G0 + KG] for k in ks], 0)))
           for n_, SS_, ks in (("36", JS_DS, KEYS36), ("37", JS_DS, list(JS_DS)))}
    rows = []
    for sc in scen:
        pools = [np.where((lab == sc) & (hk == h))[0] for h in range(len(KEYS36))]; pools = [p for p in pools if len(p)]
        rr = np.random.default_rng(1); C = []
        for _ in range(2000):
            idx = [p[rr.integers(len(p))] for p in pools]
            C.append(np.exp(np.median(np.mean([DR[i][rr.integers(0, len(DR[i]), 1000)] for i in idx], 0))))
        C = np.array(C)
        rows.append(dict(scenario=sc, cohort_median=np.median(C), pct5=np.percentile(C, 5), pct95=np.percentile(C, 95),
                         p_data36=np.mean(C <= obs["36"]), p_data37=np.mean(C <= obs["37"]),
                         **{f"mean_d_{p}": MU[lab == sc, j].mean() for j, p in enumerate(JP) if p in ("w_GS", "sig_slow", "log10_sig")}))
    T2 = pd.DataFrame(rows)
    MU36 = np.array([JS_DS[k][:, G0:].mean(0) for k in KEYS36]).mean(0)
    print(f"\ndata cohort-mean w_GS factor: in reach ({len(KEYS36)}) ×{obs['36']:.3f}, all paired ({len(JS_DS)}) ×{obs['37']:.3f}; "
          f"data mean Δ (in reach): w_GS {MU36[KG]:+.3f}, sig_slow {MU36[JP.index('sig_slow')]:+.3f}, log10_sig {MU36[JP.index('log10_sig')]:+.3f}")
    print(f"calibrated null test (Table 5), data and simulated cohorts on the same {len(KEYS36)} hemispheres "
          "(p_data36: data on the in-reach set; p_data37: data on all paired hemispheres, as before):")
    print(T2.round(3).to_string(index=False)); T2.to_csv(R13 / "B_calibrated_null_36.csv", index=False); SUMMARY["B_null"] = T2


run(B_consistency)

# %% [markdown]
# ## C. Precise calibration of the hierarchical estimator
#
# Each cohort has the data's structure: the 36 in-reach hemispheres of 19 patients, OFF states from their posteriors,
# true change Δ_ij = ln(factor) + u_i + ε_ij (u_i: patient effect, normal or Student-t₃ scaled to s.d. τ; ε_ij: s.d.
# SIGMA_TRUE), session nuisance as in training, no other candidate change. Designs are simulated one at a time and
# saved, so an interrupted run resumes. Reported with 95% Monte Carlo (Wilson / bootstrap) intervals.

# %%
@part("C")
def C_simulate():
    for di, (lab, fac, dist, tau, nco) in enumerate(C_DESIGN):
        f = R13 / f"C_design_{di}.csv"
        if f.exists():
            print(f"[{lab}] loaded"); continue
        n_c = max(2, int(round(nco * COHORT_SCALE))); rng = np.random.default_rng(SEED4 + 1000 * di)
        TO, TN, meta = [], [], []
        for c in range(n_c):
            z = rng.standard_t(3, len(SUBS)) / np.sqrt(3) if dist == "t3" else rng.standard_normal(len(SUBS))
            u = dict(zip(SUBS, tau * z))
            for h, k in enumerate(KEYS36):
                d = float(np.clip(np.log(fac) + u[k[0]] + SIGMA_TRUE * rng.standard_normal(), -0.95 * WG, 0.95 * WG))
                th = off_draw(rng, k); Dm = np.zeros((1, nM)); Dm[0, KG] = d
                TO.append(th); TN.append(apply_full(th, Dm, nuisance_draw(th, rng))); meta.append((c, h, d))
        TO, TN = np.concatenate(TO), np.concatenate(TN)
        print(f"[{lab}] simulating {n_c} cohorts ({len(TO)} pairs)")
        D = posterior_changes(TO, TN, SEED4 + 10 + di, n_samp=2000)
        rows = []
        for c in range(n_c):
            idx = [i for i, m in enumerate(meta) if m[0] == c and D[i] is not None]
            Ds = [D[i][:, KG] for i in idx]; subj = SUBJ[[meta[i][1] for i in idx]]
            r_ = np.random.default_rng(c); G = np.mean([d[r_.integers(0, len(d), 4000)] for d in Ds], 0)
            mv = np.array([flat_likelihood(d, WG) for d in Ds])
            h, _ = hierarchical_normal(mv[:, 0], mv[:, 1], subj, WG)
            rows.append(dict(design=lab, factor=fac, dist=dist, tau_true=tau, cohort=c, true_mu=np.log(fac),
                             true_sample_mean=np.mean([meta[i][2] for i in idx]),
                             sparse_est=np.median(G), sparse_lo=np.percentile(G, 5), sparse_hi=np.percentile(G, 95), sparse_P=(G < 0).mean(),
                             hier_est=h["mu_med"], hier_lo=h["mu_lo"], hier_hi=h["mu_hi"], hier_P=h["P_mu_neg"], hier_tau=h["tau_med"]))
        pd.DataFrame(rows).to_csv(f, index=False)


@part("C")
def C_summarise():
    T = pd.concat([pd.read_csv(R13 / f"C_design_{di}.csv") for di in range(len(C_DESIGN)) if (R13 / f"C_design_{di}.csv").exists()])
    T.to_csv(R13 / "C_cohorts_all.csv", index=False)
    rows = []
    for lab, g in T.groupby("design", sort=False):
        n = len(g)
        cov_h = int(((g.hier_lo <= g.true_mu) & (g.true_mu <= g.hier_hi)).sum())
        cov_s = int(((g.sparse_lo <= g.true_sample_mean) & (g.true_sample_mean <= g.sparse_hi)).sum())
        fp_h = int((g.hier_P > .95).sum()); fp_s = int((g.sparse_P > .95).sum())
        b = g.hier_est - g.true_mu; bs = g.sparse_est - g.true_sample_mean
        rows.append(dict(design=lab, n_cohorts=n, true=np.exp(g.true_mu.iloc[0]),
                         hier_mean=np.exp(g.hier_est.mean()), hier_bias=b.mean(), hier_bias_se=b.std(ddof=1) / np.sqrt(n),
                         hier_sd=g.hier_est.std(ddof=1), hier_cov90=cov_h / n, hier_cov90_ci=wilson(cov_h, n),
                         hier_P95=fp_h / n, hier_P95_ci=wilson(fp_h, n),
                         sparse_mean=np.exp(g.sparse_est.mean()), sparse_bias=bs.mean(), sparse_cov90=cov_s / n,
                         sparse_P95=fp_s / n, sparse_P95_ci=wilson(fp_s, n)))
    S_ = pd.DataFrame(rows)
    fmt = lambda t: f"[{t[0]:.2f}, {t[1]:.2f}]"
    show = S_.copy()
    for c in ("hier_cov90_ci", "hier_P95_ci", "sparse_P95_ci"):
        show[c] = show[c].map(fmt)
    print("Hierarchical estimator vs sparse-prior cohort mean on simulated cohorts (log-scale bias ± s.e.; 95% Wilson intervals;\n"
          "P95 = fraction of cohorts with P(decrease) > 0.95: the false-positive rate when the true factor is ×1.00)")
    print(show.round(3).to_string(index=False)); S_.to_csv(R13 / "C_summary.csv", index=False); SUMMARY["C"] = show

    # data estimate (36 hemispheres, STN+MEG network)
    Ds = [JS_DS[k][:, G0 + KG] for k in KEYS36]; mv = np.array([flat_likelihood(d, WG) for d in Ds])
    hd, _ = hierarchical_normal(mv[:, 0], mv[:, 1], SUBJ, WG); est = hd["mu_med"]
    # (1) data vs null distribution of estimates
    for lab in [d[0] for d in C_DESIGN if d[1] == 1.0]:
        g = T[T.design == lab]
        if len(g):
            k = int((g.hier_est <= est).sum())
            print(f"null design {lab}: {k}/{len(g)} null cohorts with an estimate ≤ the data's (×{np.exp(est):.3f}); "
                  f"null estimates ×{np.exp(g.hier_est.mean()):.3f} ± {g.hier_est.std(ddof=1):.3f} (log s.d.); "
                  f"z = {(est - g.hier_est.mean()) / g.hier_est.std(ddof=1):+.1f}")
    # (2) calibration curve and Neyman belt (normal patient effects, τ = 0.09)
    base = T[(T.dist == "normal") & (T.tau_true == 0.09)]
    def belt(dfb):
        mus = np.sort(dfb.true_mu.unique())
        m = np.array([dfb[dfb.true_mu == u].hier_est.mean() for u in mus]); s = np.array([dfb[dfb.true_mu == u].hier_est.std(ddof=1) for u in mus])
        a1, a0 = np.polyfit(mus, m, 1); s1, s0 = np.polyfit(mus, s, 1)
        grid = np.linspace(np.log(0.5), np.log(1.1), 1201)
        mg, sg = a0 + a1 * grid, np.maximum(s0 + s1 * grid, 1e-3)
        inside = np.abs(est - mg) <= 1.645 * sg
        point = (est - a0) / a1
        return point, (grid[inside].min() if inside.any() else np.nan), (grid[inside].max() if inside.any() else np.nan), a0, a1
    point, lo, hi, a0, a1 = belt(base)
    rb = np.random.default_rng(SEED4 + 3); P_ = []
    for _ in range(500):
        bb = pd.concat([g.sample(len(g), replace=True, random_state=int(rb.integers(1e9))) for _, g in base.groupby("true_mu")])
        P_.append(belt(bb)[0])
    print(f"\ncalibration (normal patient effects, τ = 0.09): E[estimate] = {a0:+.3f} + {a1:.3f}·μ (log scale)")
    print(f"data hierarchical estimate ×{np.exp(est):.3f} [{np.exp(hd['mu_lo']):.3f}, {np.exp(hd['mu_hi']):.3f}] ({len(KEYS36)} hemispheres)")
    print(f"bias-corrected population change ×{np.exp(point):.3f} (calibration-curve bootstrap 90%: ×{np.exp(np.percentile(P_, 5)):.3f}–"
          f"×{np.exp(np.percentile(P_, 95)):.3f}); Neyman-belt 90% confidence interval ×{np.exp(lo):.3f}–×{np.exp(hi):.3f}")
    SUMMARY["C_data"] = dict(est=np.exp(est), corrected=np.exp(point), boot=(np.exp(np.percentile(P_, 5)), np.exp(np.percentile(P_, 95))),
                             neyman=(np.exp(lo), np.exp(hi)))
    pd.DataFrame([SUMMARY["C_data"]]).to_csv(R13 / "C_data_calibrated.csv", index=False)
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(7.48, 3.1))
    for est_c, col, lab, x_off in (("sparse_est", "#8c8c8c", "sparse-prior cohort mean", -0.006), ("hier_est", "#2a78d6", "hierarchical", 0.006)):
        g = base.groupby("true_mu")[est_c].agg(["mean", lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
        g.columns = ["m", "lo", "hi"]; x = np.exp(g.index.values) + x_off
        axs[0].errorbar(x, np.exp(g.m), yerr=[np.exp(g.m) - np.exp(g.lo), np.exp(g.hi) - np.exp(g.m)], fmt="o", color=col, ms=4, lw=.8, label=lab)
    xx = np.linspace(.65, 1.03, 50); axs[0].plot(xx, xx, "k--", lw=.6)
    axs[0].axhline(np.exp(est), color="#eb6834", lw=.8, ls=":", label="data (hierarchical)")
    axs[0].set_xlabel(r"true population $w_{GS}$ ON/OFF"); axs[0].set_ylabel("estimate (mean, 5–95% of cohorts)"); axs[0].legend(frameon=False, fontsize=6)
    S2 = S_.set_index("design")
    labs = [d[0] for d in C_DESIGN if d[0] in S2.index]; x = np.arange(len(labs))
    cov = S2.loc[labs, "hier_cov90"].values; ci = np.array([S2.loc[l, "hier_cov90_ci"] for l in labs])
    fp = S2.loc[labs, "hier_P95"].values; cf = np.array([S2.loc[l, "hier_P95_ci"] for l in labs])
    axs[1].errorbar(x - .12, cov, yerr=[cov - ci[:, 0], ci[:, 1] - cov], fmt="o", color="#2a78d6", ms=4, lw=.8, label="90% coverage")
    axs[1].errorbar(x + .12, fp, yerr=[fp - cf[:, 0], cf[:, 1] - fp], fmt="s", color="#eb6834", ms=4, lw=.8, label="P(decrease) > 0.95")
    axs[1].axhline(.9, color="k", lw=.5, ls=":"); axs[1].set_ylim(-.02, 1.02)
    axs[1].set_xticks(x, labs, rotation=35, ha="right", fontsize=6.5); axs[1].legend(frameon=False, fontsize=6)
    axs[1].set_title("hierarchical estimator (95% Monte Carlo intervals)", fontsize=8)
    fig.tight_layout(); fig.savefig(REVF / "R13_hierarchical_calibration.png", dpi=300); plt.show()


run(C_simulate)
run(C_summarise)

# %% [markdown]
# ## D. Stability of the signature comparison (6 → 12 pairs per hemisphere)

# %%
SINGLE = ["nuisance only", "noise ↓", "slow input ↓", "w_GS ↓"]; COMB = "w_GS ↓ + noise ↓"; SCEN = SINGLE + [COMB]
SIGS = {"β-route (6)": ["w_GS", "sig_slow", "log10_sig", "g_ctx", "k_d", "w_TC"], "all 11": list(JP)}


def scen_delta(sc, rng):
    Dm = np.zeros((1, nM))
    if sc == "noise ↓":
        Dm[0, JP.index("log10_sig")] = -rng.uniform(.05, .25)
    elif sc == "slow input ↓":
        Dm[0, JP.index("sig_slow")] = np.log(rng.uniform(.3, .85))
    elif sc == "w_GS ↓":
        Dm[0, KG] = np.log(rng.uniform(.6, .95))
    elif sc == COMB:
        Dm[0, KG] = np.log(rng.uniform(.6, .95)); Dm[0, JP.index("log10_sig")] = -rng.uniform(.02, .20)
    return Dm


@part("D")
def D_simulate():
    base_c = dict(CIRC)
    for ci, cond in enumerate(R12_CONDITIONS):
        f = R13 / f"D_extra_{ci}.npz"
        if f.exists():
            print(f"[{cond}] extra pairs loaded"); continue
        rng = np.random.default_rng(SEED4 + 7000 + 100 * ci); TO, TN, lab, hk, half = [], [], [], [], []
        for sc in SCEN:
            for h, k in enumerate(KEYS36):
                for r_ in range(K_NEW):
                    th = off_draw(rng, k); TO.append(th); TN.append(apply_full(th, scen_delta(sc, rng), nuisance_draw(th, rng)))
                    lab.append(sc); hk.append(h); half.append(r_ < K_NEW // 2)
        TO, TN = np.concatenate(TO), np.concatenate(TN)
        try:
            for kk, v in MISSPEC[cond].items():
                CIRC[kk] = base_c[kk] * v
            print(f"[{cond}] simulating {len(TO)} extra pairs")
            D = posterior_changes(TO, TN, SEED4 + 8000 + ci)
        finally:
            CIRC.clear(); CIRC.update(base_c)
        ok = np.array([d is not None for d in D])
        MU = np.array([d.mean(0) for d in D if d is not None])
        np.savez(f, MU=MU, lab=np.array(lab)[ok], hk=np.array(hk)[ok], half=np.array(half)[ok])


def load_pairs(ci, with_extra):
    a = np.load(REV / f"R12_pairs_{ci}.npz", allow_pickle=True); b = np.load(REV / f"R12b_combined_{ci}.npz", allow_pickle=True)
    parts_ = [a, b] + ([np.load(R13 / f"D_extra_{ci}.npz", allow_pickle=True)] if with_extra else [])
    return dict(MU=np.concatenate([p["MU"] for p in parts_]), lab=np.concatenate([p["lab"].astype(str) for p in parts_]),
                hk=np.concatenate([p["hk"] for p in parts_]), half=np.concatenate([p["half"].astype(bool) for p in parts_]))


def cohorts(P, cols, half, n, rng):
    X, y = [], []
    for si, sc in enumerate(SCEN):
        pools = [np.where((P["lab"] == sc) & (P["hk"] == h) & (P["half"] == half))[0] for h in range(len(KEYS36))]
        pools = [p for p in pools if len(p)]
        for _ in range(n):
            X.append(np.mean([P["MU"][p[rng.integers(len(p))]][cols] for p in pools], 0)); y.append(si)
    return np.array(X), np.array(y)


@part("D")
def D_compare():
    MU_dat = np.array([JS_DS[k][:, G0:].mean(0) for k in KEYS36])
    rows = []
    for sig, dims in SIGS.items():
        cols = [JP.index(d) for d in dims]; sig_dat = MU_dat[:, cols].mean(0)
        rb = np.random.default_rng(SEED4 + 9); SB = []
        for _ in range(N_BOOT):
            ss = rb.choice(SUBS, len(SUBS)); idx = np.concatenate([np.where(SUBJ == s_)[0] for s_ in ss]); SB.append(MU_dat[idx][:, cols].mean(0))
        SB = np.array(SB)
        for ci, cond in enumerate(R12_CONDITIONS):
            for K, extra in ((K_OLD, False), (K_OLD + K_NEW, True)):
                P = load_pairs(ci, extra); rng = np.random.default_rng(SEED4 + 7)
                Xtr, ytr = cohorts(P, cols, True, 600, rng); Xte, yte = cohorts(P, cols, False, 300, rng)
                mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-12
                clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit((Xtr - mu_) / sd_, ytr)
                acc = np.mean(clf.predict((Xte - mu_) / sd_) == yte); PB = clf.predict_proba((SB - mu_) / sd_)
                for si, sc in enumerate(SCEN):
                    Xs = (Xte[yte == si] - mu_) / sd_; m_ = Xs.mean(0)
                    ci_ = np.linalg.pinv(LedoitWolf().fit((Xtr[ytr == si] - mu_) / sd_).covariance_)
                    dm = lambda x: float((x - m_) @ ci_ @ (x - m_))
                    d_dat = dm((sig_dat - mu_) / sd_)
                    rows.append(dict(signature=sig, condition=cond, K=K, scenario=sc, accuracy=acc,
                                     frac_most_probable=np.mean(PB.argmax(1) == si), typicality_p=np.mean([dm(x) >= d_dat for x in Xs]),
                                     cohort_sd_wGS=Xte[yte == si][:, 0].std() if dims[0] == "w_GS" else np.nan))
    T = pd.DataFrame(rows); T.to_csv(R13 / "D_signature_stability.csv", index=False)
    pd.set_option("display.width", 250)
    for sig in SIGS:
        t = T[T.signature == sig]
        print(f"\n===== {sig}: fraction of patient bootstraps in which each scenario is the most probable (K = pairs per hemisphere) =====")
        pv = t.pivot_table(index="scenario", columns=["condition", "K"], values="frac_most_probable").loc[SCEN, R12_CONDITIONS]
        print(pv.round(3).to_string())
        inc = t[t.scenario.isin(["w_GS ↓", COMB])].groupby(["condition", "K"]).frac_most_probable.sum().unstack().loc[R12_CONDITIONS]
        print("scenarios containing w_GS ↓:"); print(inc.round(3).to_string())
        print("held-out accuracy:"); print(t.groupby(["condition", "K"]).accuracy.first().unstack().loc[R12_CONDITIONS].round(3).to_string())
        print("typicality p of the data:")
        print(t.pivot_table(index="scenario", columns=["condition", "K"], values="typicality_p").loc[SCEN, R12_CONDITIONS].round(3).to_string())
        if sig == list(SIGS)[0]:
            print("s.d. of the simulated cohort w_GS signature (spread of the reference distribution):")
            print(t.pivot_table(index="scenario", columns=["condition", "K"], values="cohort_sd_wGS").loc[SCEN, R12_CONDITIONS].round(4).to_string())
    SUMMARY["D"] = T


run(D_simulate)
run(D_compare)

# %% [markdown]
# ## Summary

# %%
for k, v in SUMMARY.items():
    if k in ("B", "B_null", "C", "C_data"):
        print(f"\n=== {k} ===")
        print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
print(f"\ntables: {R13}\nfigures: {REVF}")

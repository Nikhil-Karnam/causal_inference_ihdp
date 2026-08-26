import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as t_dist, wilcoxon
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from models import dumpsterfire, run_dragonnet, xlearner, xlearner_oof


# rows = 672, covariates = 25
train_data = dict(np.load("ihdp_npci_1-100.train.npz"))
test_data = dict(np.load("ihdp_npci_1-100.test.npz"))

IHDP_NAMES = [
    "bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
    "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
    "cig", "first", "booze", "drugs", "work.dur", "prenatal",
    "ark", "ein", "har", "mia", "pen", "tex", "was",
]

# ------------------------------------ helper defs ------------------------------------

# outputs confidence interval as formatted string
def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def vein(est, ses):
    est = np.asarray(est, float)
    ses = np.maximum(np.asarray(ses, float), 1e-12)
    z = norm.ppf(0.975)
    p = min(1, 2 * np.median(2 * norm.sf(abs(est / ses))))
    return (np.median(est), np.median(ses),
            np.median(est - z * ses), np.median(est + z * ses), p)


def vein_table(title, key, labels, est, ses):
    rows = [[label, e, ci(lo, hi), p]
            for label, (e, s, lo, hi, p) in zip(labels, map(vein, est, ses))]

    print(f"\n**{title}**")
    print(tabulate(rows,
                   headers=[key, "median", "95% CI", "p"],
                   floatfmt=("", ".3f", "", ".3g"),
                   colalign=("left",) * 4,
                   tablefmt="github"))


def hist_grid(title, labels, series, filename, cols=3):
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)

    for ax, label, v in zip(axes.flat, labels, series):
        ax.hist(v, bins=25, color="steelblue")
        ax.axvline(np.median(v), color="black")
        ax.axvline(0, color="crimson", linestyle="--")
        ax.set_title(f"{label}   median {np.median(v):.2f}", fontsize=9)

    for ax in axes.flat[len(labels):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

# ------------------------------------ analyses ------------------------------------

def model_accuracy(base, xl, reg, noreg, cates_oracle):
    def get_pehes(cates):
        return np.array([np.sqrt(np.mean((c - o) ** 2)) 
                         for c, o in zip(cates, cates_oracle)])

    def stats(pehes):
        se = pehes.std(ddof=1) / np.sqrt(len(pehes))
        return [pehes.mean(),
                np.median(pehes),
                ci(pehes.mean() - 1.96 * se, pehes.mean() + 1.96 * se)]

    def stats_compare(i, j):
        # hodges lehmann estimate
        d = i - j
        m = len(d)
        walsh = np.sort(((d[:, None] + d) / 2)[np.triu_indices(m)])
        k = np.clip(int(m * (m + 1) / 4 - 1.96 * np.sqrt(m * (m + 1) * (2 * m + 1) / 24)),
                    0, (len(walsh) - 1) // 2)

        return [np.median(walsh),
                ci(walsh[k], walsh[-k - 1]),
                wilcoxon(i, j).pvalue,
                np.median(100 * (1 - i / j))]

    b, x, r, n = (get_pehes(m) for m in (base, xl, reg, noreg))

    names = ["baseline", "xlearner", "dragonnet reg", "dragonnet no-reg"]
    pairs = [("xl - base", x, b),
             ("reg - base", r, b),
             ("no-reg - base", n, b),
             ("no-reg - reg", n, r)]

    print("\n" + tabulate([[name] + stats(pehes) for name, pehes in zip(names, (b, x, r, n))],
                          headers=["model", "mean", "median", "95% CI"],
                          floatfmt=".3f",
                          colalign=("left",) * 4,
                          tablefmt="github"))

    print("\n" + tabulate([[label] + stats_compare(i, j) for label, i, j in pairs],
                          headers=["comparison", "lehmann", "95% CI", "p (wilcoxon)", "% reduction"],
                          floatfmt=("", ".3f", "", ".3g", ".1f"),
                          colalign=("left",) * 5,
                          tablefmt="github"))

    hist_grid("pehe per rep", names, [b, x, r, n], "hist_pehe.png", cols=4)


def blp(t, y, cates, pred_t, pred_y0):
    est, ses = [], []

    for t_i, y_i, c_i, pt_i, py0_i in zip(t, y, cates, pred_t, pred_y0):
        t_residual = t_i - pt_i
        cols = pd.DataFrame({'pred_y0': py0_i,
                             'b1': t_residual,
                             'b2': t_residual * (c_i - c_i.mean())})

        fit = sm.WLS(y_i, sm.add_constant(cols),
                     weights=1 / (pt_i * (1 - pt_i))).fit(cov_type='HC3')

        est.append(fit.params[['b1', 'b2']])
        ses.append(fit.bse[['b1', 'b2']])

    vein_table("blp", "term", ["ate (b1)", "het (b2)"],
               np.array(est).T, np.array(ses).T)


def gates(t, y, cates, pred_t, pred_y0):
    gammas = [f'gamma{k+1}' for k in range(5)]
    est, ses = [], []

    for t_i, y_i, c_i, pt_i, py0_i in zip(t, y, cates, pred_t, pred_y0):
        #replacing with ranks prevents issue where qcut fails due to same numbers
        g = pd.qcut(pd.Series(c_i).rank(method='first'), 5, labels=False).to_numpy()
        d = t_i - pt_i
        cols = pd.DataFrame({'pred_y0': py0_i,
                             **{gammas[k]: d * (g == k) for k in range(5)}})

        fit = sm.WLS(y_i, sm.add_constant(cols),
                     weights=1 / (pt_i * (1 - pt_i))).fit(cov_type='HC3')
        contrast = fit.t_test('gamma5 - gamma1')

        est.append(list(fit.params[gammas]) + [float(np.squeeze(contrast.effect))])
        ses.append(list(fit.bse[gammas]) + [float(np.squeeze(contrast.sd))])

    est, ses = np.array(est).T, np.array(ses).T
    labels = [f"group {k+1}" for k in range(5)] + ["gamma5-gamma1"]

    vein_table("gates", "group", labels, est, ses)
    hist_grid("gates per rep", labels, est, "hist_gates.png")


def gates_oracle(cates_oracle):
    est, ses = [], []

    for c in cates_oracle:
        g = pd.qcut(pd.Series(c).rank(method='first'), 5, labels=False).to_numpy()
        est.append([c[g == k].mean() for k in range(5)])
        ses.append([c[g == k].std(ddof=1) / np.sqrt((g == k).sum()) for k in range(5)])

    vein_table("gates oracle", "group", [f"group {k+1}" for k in range(5)],
               np.array(est).T, np.array(ses).T)


def clan(x, cates, names=IHDP_NAMES):
    g1_m, g1_v, g5_m, g5_v = [], [], [], []

    for x_i, c_i in zip(x, cates):
        g = pd.qcut(pd.Series(c_i).rank(method='first'), 5, labels=False).to_numpy()
        low, high = x_i[g == 0], x_i[g == 4]

        g1_m.append(low.mean(0))
        g5_m.append(high.mean(0))
        g1_v.append(low.var(0, ddof=1) / len(low))
        g5_v.append(high.var(0, ddof=1) / len(high))

    diff = (np.array(g5_m) - np.array(g1_m)).T
    diff_se = np.sqrt(np.array(g1_v) + np.array(g5_v)).T

    stats = [vein(d, s) for d, s in zip(diff, diff_se)]
    p_adj = multipletests([s[4] for s in stats], method='holm')[1]

    rows = [[name, e, ci(lo, hi), p, pa]
            for name, (e, s, lo, hi, p), pa in zip(names, stats, p_adj)]

    print("\n**clan**")
    print(tabulate(rows,
                   headers=["covariate", "median diff", "95% CI", "p", "p adj (holm)"],
                   floatfmt=("", ".3f", "", ".3g", ".3g"),
                   colalign=("left",) * 5,
                   tablefmt="github"))

    flagged = np.where(p_adj < 0.05)[0]
    if len(flagged):
        hist_grid("clan diff per rep", [names[j] for j in flagged],
                  [diff[j] for j in flagged], "hist_clan.png")


def qini(t, y, cates, pred_t, pred_y0, pred_y1, cates_oracle, n_points=101):
    observed_curves, oracle_curves = [], []

    for t_i, y_i, c_i, pt_i, py0_i, py1_i, co_i in zip(t, y, cates, pred_t, pred_y0, pred_y1, cates_oracle):
        n = len(c_i)
        ks = np.round(np.linspace(0, 1, n_points) * n).astype(int)

        order = np.argsort(-c_i)
        t_o, y_o, p_o = t_i[order], y_i[order], pt_i[order]
        y0_o, y1_o = py0_i[order], py1_i[order]

        scores = (y1_o - y0_o
                  + t_o * (y_o - y1_o) / p_o
                  - (1 - t_o) * (y_o - y0_o) / (1 - p_o))
        observed_curves.append(np.concatenate([[0], np.cumsum(scores)])[ks])

        n_o = len(co_i)
        ks_o = np.round(np.linspace(0, 1, n_points) * n_o).astype(int)
        oracle_curves.append(np.concatenate([[0], np.cumsum(co_i[np.argsort(-co_i)])])[ks_o])

    observed = np.array(observed_curves).mean(axis=0)
    oracle = np.array(oracle_curves).mean(axis=0)
    fractions = np.linspace(0, 1, len(observed))

    area_obs = np.trapezoid(observed - fractions * observed[-1], fractions)
    area_orc = np.trapezoid(oracle - fractions * oracle[-1], fractions)

    plt.figure(figsize=(7, 5))
    plt.plot(fractions, observed, label=f"x-learner (area {area_obs:.3f})")
    plt.plot(fractions, oracle, label=f"perfect ranking (area {area_orc:.3f})", linestyle="--")
    plt.plot(fractions, fractions * observed[-1], color="gray", linewidth=1, label="random")
    plt.xlabel("fraction treated")
    plt.ylabel("cumulative gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qini.png", dpi=150)

# ------------------------------------------------------------------------

def run_rep(i):
    t = train_data["t"][:, i]
    y = train_data["yf"][:, i]
    x = train_data["x"][:, :, i]

    t_test = test_data["t"][:, i]
    y_test = test_data["yf"][:, i]
    x_test = test_data["x"][:, :, i]

    cates_oof, pred_t_oof, pred_y0_oof, pred_y1_oof = xlearner_oof(t, y, x)

    return {
        "t": t, "y": y, "x": x,
        "cates_oracle_train": train_data["mu1"][:, i] - train_data["mu0"][:, i],
        "cates_oracle_test": test_data["mu1"][:, i] - test_data["mu0"][:, i],
        "cate_dumpsterfire": dumpsterfire(t_test, y_test),
        "cates_xl": xlearner(t, y, x, x_test)[0],
        "cates_oof": cates_oof,
        "pred_t_oof": np.clip(pred_t_oof, 0.05, 0.95),
        "pred_y0_oof": pred_y0_oof,
        "pred_y1_oof": pred_y1_oof,
        "cates_dn_reg": run_dragonnet(t, y, x, x_test, i, True),
        "cates_dn_noreg": run_dragonnet(t, y, x, x_test, i, False),
    }


if __name__ == '__main__':
    reps = Parallel(n_jobs=-1, verbose=10)(delayed(run_rep)(i) for i in range(100))
    r = {k: [rep[k] for rep in reps] for k in reps[0]}

    qini(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"], r["pred_y1_oof"], r["cates_oracle_train"])
    model_accuracy(r["cate_dumpsterfire"], r["cates_xl"], r["cates_dn_reg"], r["cates_dn_noreg"], r["cates_oracle_test"])
    blp(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"])
    gates(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"])
    gates_oracle(r["cates_oracle_train"])
    clan(r["x"], r["cates_oof"])
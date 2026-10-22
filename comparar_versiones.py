import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta3", default="resultados/01_beta3_casos_a_b_57130/figuras/resumen_caso_a.csv")
    ap.add_argument("--beta2", default="resultados/03_beta2_caso_a_58167/figuras/resumen_caso_a.csv")
    ap.add_argument("--out", default="resultados/04_comparacion_versiones")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    b3 = pd.read_csv(a.beta3)
    b2 = pd.read_csv(a.beta2)
    cols = ["n", "p", "Ts", "t_total", "S", "E", "acc"]
    m = b2[cols].merge(b3[cols], on=["n", "p"], suffixes=("_beta2", "_beta3"))
    m.to_csv(os.path.join(a.out, "comparacion_beta2_beta3.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, c in zip(sorted(m.n.unique()), ["C0", "C1", "C2", "C3"]):
        g = m[m.n == n]
        axs[0].plot(g.p, g.S_beta3, "o-", color=c, label=f"beta3 n={n}")
        axs[0].plot(g.p, g.S_beta2, "s--", color=c, label=f"beta2 n={n}")
        axs[1].plot(g.p, g.E_beta3, "o-", color=c)
        axs[1].plot(g.p, g.E_beta2, "s--", color=c)
    ps = sorted(m.p.unique())
    axs[0].plot(ps, ps, "k:", alpha=0.5, label="ideal S=p")
    axs[0].set_xscale("log", base=2); axs[0].set_yscale("log", base=2); axs[0].set_ylabel("speedup S = T_s / T_p")
    axs[1].set_xscale("log", base=2); axs[1].set_ylim(0, 1.15); axs[1].set_ylabel("eficiencia E = S / p")
    axs[1].axhline(1, color="k", ls=":", alpha=0.5)
    for ax in axs:
        ax.set_xlabel("p"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Caso A (q=32): beta2 ingenua (batch completo, --) vs beta3 optimizada (bloques, —)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparacion_beta2_beta3.png"), dpi=160)
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

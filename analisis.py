import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _matriz(df):
    return np.column_stack([np.ceil(df.q / df.p) * df.n, np.log2(df.p), np.ones(len(df))])


def ajustar(df):
    coef, *_ = np.linalg.lstsq(_matriz(df), df.t_total.to_numpy(), rcond=None)
    return coef


def r2(df, coef):
    y = df.t_total.to_numpy()
    pred = _matriz(df) @ coef
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def teoria(coef, n, q, p):
    a, b, c = coef
    return a * math.ceil(q / p) * n + b * math.log2(p) + c


def metricas(med, exp):
    de = med[med.experimento == exp]
    if not len(de):
        return None
    seq = de[de.version == "beta1"].set_index(["n", "q"]).t_total
    par = de[de.version != "beta1"].copy()
    par["Ts"] = [seq[(n, q)] for n, q in zip(par.n, par.q)]
    par["S"] = par.Ts / par.t_total
    par["E"] = par.S / par.p
    par["flops_s"] = par.q * par.flops_modelo / par.t_total
    return par.sort_values(["n", "p"])


def _guardar(fig, out, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre), dpi=160)
    plt.close(fig)


def graficas_caso_a(A, coef, rA, out):
    ns = sorted(A.n.unique())
    ps = sorted(A.p.unique())
    pf = np.arange(min(ps), max(ps) + 1)
    q = int(A.q.iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n]
        l, = ax.plot(g.p, g.t_total, "o-", label=f"n={n} medido")
        ax.plot(pf, [teoria(coef, n, q, x) for x in pf], "--", color=l.get_color(), alpha=0.7)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("p (procesos)"); ax.set_ylabel("tiempo (s)")
    ax.set_title(f"Caso A: T_p medido vs teórico ajustado, q={q}, R²={rA:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_a_tiempos_vs_p.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in ps:
        g = A[A.p == p].sort_values("n")
        l, = ax.plot(g.n, g.t_total, "o-", label=f"p={p}")
        ax.plot(g.n, [teoria(coef, n, q, p) for n in g.n], "--", color=l.get_color(), alpha=0.7)
    ax.set_xlabel("n (muestras)"); ax.set_ylabel("tiempo (s)"); ax.set_title(f"Caso A: T_p vs n, q={q} (punteado: modelo ajustado)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_a_tiempos_vs_n.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n]
        ax.plot(g.p, g.S, "o-", label=f"n={n}")
    ax.plot(ps, ps, "k--", alpha=0.5, label="ideal S=p")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("p"); ax.set_ylabel("speedup S = T_s / T_p"); ax.set_title(f"Caso A: speedup, q={q}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_a_speedup.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n]
        ax.plot(g.p, g.E, "o-", label=f"n={n}")
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15)
    ax.set_xlabel("p"); ax.set_ylabel("eficiencia E = S / p"); ax.set_title(f"Caso A: escalabilidad fuerte (q={q} fijo)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_a_eficiencia.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = A[A.n == n]
        ax.plot(g.p, g.flops_s / 1e9, "o-", label=f"n={n}")
    ax.set_xscale("log", base=2); ax.set_xlabel("p"); ax.set_ylabel("GFLOP/s"); ax.set_title("Caso A: FLOPs vs p")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_a_flops_vs_p.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    g = A[A.n == max(ns)]
    ax.bar([str(p) for p in g.p], g.t_map, label="map (entrenar)")
    ax.bar([str(p) for p in g.p], g.t_total - g.t_map, bottom=g.t_map, label="reduce + arranque de procesos")
    ax.set_xlabel("p"); ax.set_ylabel("tiempo (s)"); ax.set_title(f"Caso A: desglose del tiempo, n={max(ns)}"); ax.legend()
    _guardar(fig, out, "caso_a_desglose.png")


def graficas_caso_b(B, coef, rB, out):
    ns = sorted(B.n.unique())
    ps = sorted(B.p.unique())
    pf = np.arange(min(ps), max(ps) + 1)
    a = coef[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = B[B.n == n]
        l, = ax.plot(g.p, g.t_total, "o-", label=f"n={n} medido")
        ax.plot(pf, [teoria(coef, n, x, x) for x in pf], "--", color=l.get_color(), alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("p (= q)"); ax.set_ylabel("tiempo (s)")
    ax.set_title(f"Caso B (q = p): T_p ≈ a·n + b·log₂p + c, R²={rB:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_b_tiempos_vs_p.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ns:
        g = B[B.n == n]
        l, = ax.plot(g.p, g.E, "o-", label=f"n={n} medido")
        ax.plot(pf, [a * x * n / (x * teoria(coef, n, x, x)) for x in pf], "--", color=l.get_color(), alpha=0.7)
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15)
    ax.set_xlabel("p (= q)"); ax.set_ylabel("eficiencia E = T_s / (p·T_p)")
    ax.set_title("Caso B: escalabilidad débil (punteado: modelo ajustado)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_b_eficiencia.png")


def graficas_caso_q(Q, coef, rQ, out):
    n = int(Q.n.max())
    Q = Q[Q.n == n]
    qs = sorted(Q.q.unique())
    ps = sorted(Q.p.unique())

    E = Q.pivot(index="p", columns="q", values="E").loc[ps, qs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(E.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(len(ps)):
        for j in range(len(qs)):
            v = E.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(qs)), [str(q) for q in qs]); ax.set_yticks(range(len(ps)), [str(p) for p in ps])
    ax.set_xlabel("q (semillas)"); ax.set_ylabel("p (procesos)")
    ax.set_title(f"Eficiencia E(q, p), n={n}")
    fig.colorbar(im, ax=ax, label="E")
    _guardar(fig, out, "caso_q_eficiencia_tabla.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = Q[Q.q == q]
        ax.plot(g.p, g.E, "o-", label=f"q={q}")
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.15)
    ax.set_xlabel("p"); ax.set_ylabel("eficiencia E = S / p")
    ax.set_title(f"Escalabilidad fuerte en semillas (q fijo por curva), n={n}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_q_fuerte_eficiencia.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in qs:
        g = Q[Q.q == q]
        ax.plot(g.p, g.S, "o-", label=f"q={q}")
    ax.plot(ps, ps, "k--", alpha=0.5, label="ideal S=p")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("p"); ax.set_ylabel("speedup S = T_s / T_p")
    ax.set_title(f"Speedup por número de semillas, n={n}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_q_speedup.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in ps:
        g = Q[Q.p == p].sort_values("q")
        l, = ax.plot(g.q, g.t_total, "o-", label=f"p={p}")
        ax.plot(g.q, [teoria(coef, n, q, p) for q in g.q], "--", color=l.get_color(), alpha=0.7)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("q (semillas)"); ax.set_ylabel("tiempo (s)")
    ax.set_title(f"T_p vs q, n={n} (punteado: modelo ajustado, R²={rQ:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _guardar(fig, out, "caso_q_tiempos_vs_q.png")

    Q = Q.assign(k=Q.q // Q.p)
    ks = [k for k, g in Q.groupby("k") if len(g) >= 3]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for k in ks:
        g = Q[Q.k == k].sort_values("p")
        axs[0].plot(g.p, g.t_total, "o-", label=f"q/p={k}")
        axs[1].plot(g.p, g.E, "o-", label=f"q/p={k}")
    axs[0].set_ylabel("tiempo (s)"); axs[1].set_ylabel("eficiencia E")
    axs[1].axhline(1, color="k", ls="--", alpha=0.5); axs[1].set_ylim(0, 1.15)
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_xlabel("p"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Escalabilidad débil en semillas (q ∝ p), n={n}")
    _guardar(fig, out, "caso_q_debil.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="resultados.csv")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    raw = pd.read_csv(args.csv)
    med = raw.groupby(["experimento", "version", "n", "q", "p"], as_index=False).agg(
        t_total=("t_total", "median"), t_map=("t_map", "median"), t_reduce=("t_reduce", "median"),
        t_std=("t_total", "std"), acc=("acc", "max"), flops_modelo=("flops_modelo", "first"),
        cpu_ratio=("cpu_ratio", "min"), nivcsw=("nivcsw", "max"))

    A = metricas(med, "caso_a")
    B = metricas(med, "caso_b")
    Q = metricas(med, "caso_q")
    todos = pd.concat([x for x in (A, B, Q) if x is not None])
    coef = ajustar(todos)
    lineas = ["T_p = a*ceil(q/p)*n + b*log2(p) + c  (ajuste conjunto de los casos presentes)",
              f"a = {coef[0]:.6e} s/(modelo*muestra)", f"b = {coef[1]:.6e} s/nivel", f"c = {coef[2]:.6e} s",
              f"R^2 global = {r2(todos, coef):.5f}"]

    if A is not None:
        rA = r2(A, coef)
        lineas.append(f"R^2 caso A = {rA:.5f}")
        A["T_teo"] = [teoria(coef, n, q, p) for n, q, p in zip(A.n, A.q, A.p)]
        A.to_csv(os.path.join(args.out, "resumen_caso_a.csv"), index=False)
        graficas_caso_a(A, coef, rA, args.out)
    if B is not None:
        rB = r2(B, coef)
        lineas.append(f"R^2 caso B = {rB:.5f}")
        B["T_teo"] = [teoria(coef, n, q, p) for n, q, p in zip(B.n, B.q, B.p)]
        B.to_csv(os.path.join(args.out, "resumen_caso_b.csv"), index=False)
        graficas_caso_b(B, coef, rB, args.out)
    if Q is not None:
        rQ = r2(Q, coef)
        lineas.append(f"R^2 caso Q = {rQ:.5f}")
        Q["T_teo"] = [teoria(coef, n, q, p) for n, q, p in zip(Q.n, Q.q, Q.p)]
        Q.to_csv(os.path.join(args.out, "resumen_caso_q.csv"), index=False)
        graficas_caso_q(Q, coef, rQ, args.out)

    texto = "\n".join(lineas) + "\n"
    with open(os.path.join(args.out, "ajuste.txt"), "w") as f:
        f.write(texto)
    print(texto)
    cols = ["n", "q", "p", "t_total", "T_teo", "S", "E", "acc", "cpu_ratio", "nivcsw"]
    if A is not None:
        print("Caso A (q fijo, escalabilidad fuerte):")
        print(A[cols].to_string(index=False))
    if B is not None:
        print("\nCaso B (q = p, escalabilidad débil):")
        print(B[cols].to_string(index=False))
    if Q is not None:
        print("\nCaso Q (n fijo, variando semillas q y procesos p):")
        print(Q[cols].to_string(index=False))
    print(f"\nFiguras en {args.out}/")


if __name__ == "__main__":
    main()

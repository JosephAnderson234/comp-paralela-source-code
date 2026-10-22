import _hilos
import argparse
import csv
import os
import platform
import time

import beta1_secuencial
import beta2_pool
import beta3_pram
import mlp

CAMPOS = ["experimento", "version", "n", "q", "p", "B", "rep", "t_total", "t_map", "t_reduce", "acc", "seed", "t_cpu_map", "cpu_ratio", "nivcsw", "E", "h", "d", "flops_modelo"]


def _fila(exp, r, rep, E, h, d):
    return {**{k: r[k] for k in ("version", "n", "q", "p", "B", "t_total", "t_map", "t_reduce", "acc", "seed", "t_cpu_map", "cpu_ratio", "nivcsw")},
            "experimento": exp, "rep": rep, "E": E, "h": h, "d": d, "flops_modelo": mlp.flops_entrenamiento(r["n"], d, h, E)}


def _hora():
    return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5000,10000,20000,40000")
    ap.add_argument("--ps", default="1,2,4,8,16,32")
    ap.add_argument("--q", type=int, default=32)
    ap.add_argument("--qs", default="32,64,128,256")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--version", default="beta3", choices=["beta2", "beta3"])
    ap.add_argument("--solo", default="caso_a,caso_b")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--out", default="resultados.csv")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    ps = [int(x) for x in a.ps.split(",")]
    qs = [int(x) for x in a.qs.split(",")]
    if "caso_a" in a.solo and any(a.q % p for p in ps):
        raise SystemExit(f"q={a.q} debe ser múltiplo de todos los p={ps}")
    if "caso_q" in a.solo and any(q % p for q in qs for p in ps):
        raise SystemExit(f"cada q de {qs} debe ser múltiplo de todos los p={ps}")
    if a.pin and a.version != "beta3":
        raise SystemExit("--pin solo aplica a beta3")
    par = beta3_pram if a.version == "beta3" else beta2_pool
    extra = {"pin": True} if a.pin else {}
    d = mlp.D_DEFAULT
    B = a.B or None
    B_par = None if a.version == "beta2" else B
    nuevo = not os.path.exists(a.out)
    print(f"# {platform.node()} cpus={os.cpu_count()} q={a.q} qs={qs} E={a.E} h={a.h} d={d} B={a.B} B_par={B_par or 0} version={a.version} experimentos={a.solo} pin={a.pin}", flush=True)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        for rep in range(a.reps):
            for n in ns:
                if "caso_a" in a.solo:
                    r = beta1_secuencial.run(n, a.q, d, a.h, a.E, B)
                    w.writerow(_fila("caso_a", r, rep, a.E, a.h, d)); f.flush()
                    print(f"[{_hora()}] caso_a seq n={n} q={a.q} T={r['t_total']:.2f}s s*={r['seed']}", flush=True)
                    for p in ps:
                        r = par.run(n, a.q, p, d, a.h, a.E, B_par, **extra)
                        w.writerow(_fila("caso_a", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_a {r['version']} n={n} q={a.q} p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} cpu/pared={r['cpu_ratio']:.3f} nivcsw={r['nivcsw']}", flush=True)
                if "caso_b" in a.solo:
                    for p in ps:
                        r = beta1_secuencial.run(n, p, d, a.h, a.E, B)
                        w.writerow(_fila("caso_b", r, rep, a.E, a.h, d)); f.flush()
                        r = par.run(n, p, p, d, a.h, a.E, B_par, **extra)
                        w.writerow(_fila("caso_b", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_b {r['version']} n={n} q=p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} cpu/pared={r['cpu_ratio']:.3f} nivcsw={r['nivcsw']}", flush=True)
                if "caso_q" in a.solo:
                    for q in qs:
                        r = beta1_secuencial.run(n, q, d, a.h, a.E, B)
                        w.writerow(_fila("caso_q", r, rep, a.E, a.h, d)); f.flush()
                        print(f"[{_hora()}] caso_q seq n={n} q={q} T={r['t_total']:.2f}s s*={r['seed']}", flush=True)
                        for p in ps:
                            r = par.run(n, q, p, d, a.h, a.E, B_par, **extra)
                            w.writerow(_fila("caso_q", r, rep, a.E, a.h, d)); f.flush()
                            print(f"[{_hora()}] caso_q {r['version']} n={n} q={q} p={p} T={r['t_total']:.2f}s acc={r['acc']:.4f} s*={r['seed']} cpu/pared={r['cpu_ratio']:.3f} nivcsw={r['nivcsw']}", flush=True)


if __name__ == "__main__":
    main()

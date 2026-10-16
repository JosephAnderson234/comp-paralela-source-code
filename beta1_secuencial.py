import _hilos
import argparse
import resource
import time

import joblib

import mlp


def run(n, q, d=mlp.D_DEFAULT, h=mlp.H_DEFAULT, E=mlp.E_DEFAULT, B=None, guardar=None):
    X_tr, y_tr, X_te, y_te = mlp.cargar_datos(n, d)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = time.process_time()
    t0 = time.perf_counter()
    a_best, s_best, m_best = float("-inf"), -1, None
    for s in range(q):
        m = mlp.entrenar_mlp(s, X_tr, y_tr, h=h, E=E, B=B)
        a = mlp.accuracy(m, X_te, y_te)
        if a > a_best:
            a_best, s_best, m_best = a, s, m
    t = time.perf_counter() - t0
    c = time.process_time() - c0
    niv = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw - r0.ru_nivcsw
    if guardar:
        joblib.dump(m_best, guardar)
    return {"version": "beta1", "n": n, "q": q, "p": 1, "B": B or 0, "t_total": t, "t_map": t, "t_reduce": 0.0, "acc": a_best, "seed": s_best,
            "t_cpu_map": c, "cpu_ratio": c / t, "nivcsw": niv}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=0)
    a = ap.parse_args()
    r = run(a.n, a.q, h=a.h, E=a.E, B=a.B or None, guardar="mlp_model.joblib")
    print(f"beta1 n={r['n']} q={r['q']} B={r['B']} T={r['t_total']:.3f}s mejor semilla={r['seed']} acc={r['acc']:.4f}")

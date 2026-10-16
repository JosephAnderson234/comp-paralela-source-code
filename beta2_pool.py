import _hilos
import argparse
import multiprocessing as mp
import resource
import time
from concurrent.futures import ProcessPoolExecutor

import joblib

import mlp

_DATA = {}


def _init(X_tr, y_tr, X_te, y_te, h, E, B):
    _DATA.update(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, h=h, E=E, B=B)


def _tarea(s):
    m = mlp.entrenar_mlp(s, _DATA["X_tr"], _DATA["y_tr"], h=_DATA["h"], E=_DATA["E"], B=_DATA["B"])
    return mlp.accuracy(m, _DATA["X_te"], _DATA["y_te"]), s, m


def run(n, q, p, d=mlp.D_DEFAULT, h=mlp.H_DEFAULT, E=mlp.E_DEFAULT, B=None, guardar=None):
    X_tr, y_tr, X_te, y_te = mlp.cargar_datos(n, d)
    ctx = mp.get_context("fork")
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=p, mp_context=ctx, initializer=_init, initargs=(X_tr, y_tr, X_te, y_te, h, E, B)) as ex:
        resultados = list(ex.map(_tarea, range(q)))
    t1 = time.perf_counter()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    c = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    a_best, s_best, m_best = float("-inf"), -1, None
    for a, s, m in resultados:
        if a > a_best:
            a_best, s_best, m_best = a, s, m
    t2 = time.perf_counter()
    if guardar:
        joblib.dump(m_best, guardar)
    return {"version": "beta2", "n": n, "q": q, "p": p, "B": B or 0, "t_total": t2 - t0, "t_map": t1 - t0, "t_reduce": t2 - t1, "acc": a_best, "seed": s_best,
            "t_cpu_map": c, "cpu_ratio": c / (min(p, q) * (t1 - t0)), "nivcsw": r1.ru_nivcsw - r0.ru_nivcsw}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=0)
    a = ap.parse_args()
    r = run(a.n, a.q, a.p, h=a.h, E=a.E, B=a.B or None, guardar="mlp_model.joblib")
    print(f"beta2 n={r['n']} q={r['q']} p={r['p']} T={r['t_total']:.3f}s mejor semilla={r['seed']} acc={r['acc']:.4f}")

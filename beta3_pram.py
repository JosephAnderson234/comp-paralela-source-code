import _hilos
import argparse
import math
import os
import resource
import multiprocessing as mp
import time

import joblib
import numpy as np

import mlp


def _cpus_por_nucleo():
    primeros, hermanos, vistos = [], [], set()
    for c in sorted(os.sched_getaffinity(0)):
        base = f"/sys/devices/system/cpu/cpu{c}/topology/"
        try:
            with open(base + "physical_package_id") as f1, open(base + "core_id") as f2:
                nucleo = (f1.read().strip(), f2.read().strip())
        except OSError:
            nucleo = ("?", str(c))
        (hermanos if nucleo in vistos else primeros).append(c)
        vistos.add(nucleo)
    return primeros + hermanos


def _worker(i, p, q, h, E, B, X_tr, y_tr, X_te, y_te, A, S, W, T_map, T_red, C_map, NIV, barrera, cpu):
    if cpu >= 0:
        os.sched_setaffinity(0, {cpu})
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = time.process_time()
    t0 = time.perf_counter()
    a_loc, s_loc, m_loc = float("-inf"), -1, None
    for s in range(i, q, p):
        m = mlp.entrenar_mlp(s, X_tr, y_tr, h=h, E=E, B=B)
        a = mlp.accuracy(m, X_te, y_te)
        if a > a_loc:
            a_loc, s_loc, m_loc = a, s, m
    tam = mlp.tam_modelo(X_tr.shape[1], h)
    Wv = np.frombuffer(W, dtype=np.float64).reshape(p, tam)
    A[i] = a_loc
    S[i] = s_loc
    if m_loc is not None:
        Wv[i, :] = mlp.modelo_a_vector(m_loc)
    T_map[i] = time.perf_counter() - t0
    C_map[i] = time.process_time() - c0
    NIV[i] = resource.getrusage(resource.RUSAGE_SELF).ru_nivcsw - r0.ru_nivcsw
    barrera.wait()
    t1 = time.perf_counter()
    for d in range(math.ceil(math.log2(p)) if p > 1 else 0):
        if i < math.ceil(p / 2 ** (d + 1)):
            j = 2 * i * 2 ** d
            k = (2 * i + 1) * 2 ** d
            if k < p and A[j] < A[k]:
                A[j] = A[k]
                S[j] = S[k]
                Wv[j, :] = Wv[k, :]
        barrera.wait()
    T_red[i] = time.perf_counter() - t1


def run(n, q, p, d=mlp.D_DEFAULT, h=mlp.H_DEFAULT, E=mlp.E_DEFAULT, B=1024, guardar=None, pin=False):
    orden = _cpus_por_nucleo() if pin else None
    X_tr, y_tr, X_te, y_te = mlp.cargar_datos(n, d)
    ctx = mp.get_context("fork")
    tam = mlp.tam_modelo(d, h)
    A = ctx.Array("d", p, lock=False)
    S = ctx.Array("i", p, lock=False)
    W = ctx.RawArray("d", p * tam)
    T_map = ctx.Array("d", p, lock=False)
    T_red = ctx.Array("d", p, lock=False)
    C_map = ctx.Array("d", p, lock=False)
    NIV = ctx.Array("l", p, lock=False)
    barrera = ctx.Barrier(p)
    t0 = time.perf_counter()
    procs = [ctx.Process(target=_worker, args=(i, p, q, h, E, B, X_tr, y_tr, X_te, y_te, A, S, W, T_map, T_red, C_map, NIV, barrera, orden[i % len(orden)] if pin else -1)) for i in range(p)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    t = time.perf_counter() - t0
    m_best = mlp.vector_a_modelo(np.frombuffer(W, dtype=np.float64)[:tam], d, h, S[0])
    if guardar:
        joblib.dump(m_best, guardar)
    return {"version": "beta3_pin" if pin else "beta3", "n": n, "q": q, "p": p, "B": B or 0, "t_total": t, "t_map": max(T_map), "t_reduce": max(T_red), "acc": A[0], "seed": S[0],
            "t_cpu_map": sum(C_map), "cpu_ratio": sum(C_map) / sum(T_map), "nivcsw": sum(NIV)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--h", type=int, default=mlp.H_DEFAULT)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    r = run(a.n, a.q, a.p, h=a.h, E=a.E, B=a.B or None, guardar="mlp_model.joblib", pin=a.pin)
    print(f"{r['version']} n={r['n']} q={r['q']} p={r['p']} B={r['B']} T={r['t_total']:.3f}s (map {r['t_map']:.3f}s, reduce {r['t_reduce'] * 1e3:.3f}ms) mejor semilla={r['seed']} acc={r['acc']:.4f}")

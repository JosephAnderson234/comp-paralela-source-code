import _hilos
import argparse
import multiprocessing as mp
import os
import statistics
import time

import mlp


def _socket(cpu):
    with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as f:
        return int(f.read())


def _mhz(cpu):
    for ruta in (f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq",):
        try:
            with open(ruta) as f:
                return int(f.read()) / 1000
        except OSError:
            pass
    with open("/proc/cpuinfo") as f:
        actual = None
        for linea in f:
            if linea.startswith("processor"):
                actual = int(linea.split(":")[1])
            elif linea.startswith("cpu MHz") and actual == cpu:
                return float(linea.split(":")[1])
    return float("nan")


def _cpu_actual(pid):
    with open(f"/proc/{pid}/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[36])


def _hijo(cpu, copiar, X, y, E, B, salida):
    if cpu >= 0:
        os.sched_setaffinity(0, {cpu})
    if copiar:
        X, y = X.copy(), y.copy()
    t0 = time.perf_counter()
    c0 = time.process_time()
    mlp.entrenar_mlp(0, X, y, E=E, B=B)
    salida.put((time.perf_counter() - t0, time.process_time() - c0))


def medir(nombre, cpu, copiar, X, y, E, B, reps):
    ctx = mp.get_context("fork")
    tiempos, mhz, cpus = [], [], set()
    for _ in range(reps):
        q = ctx.Queue()
        pr = ctx.Process(target=_hijo, args=(cpu, copiar, X, y, E, B, q))
        pr.start()
        while pr.is_alive():
            try:
                c = _cpu_actual(pr.pid)
                cpus.add(c)
                mhz.append(_mhz(c))
            except (OSError, IndexError, ValueError):
                pass
            time.sleep(0.25)
        pr.join()
        pared, cpu_t = q.get()
        tiempos.append((pared, cpu_t))
    med = statistics.median(t for t, _ in tiempos)
    ratio = statistics.median(c / t for t, c in tiempos)
    socks = sorted({_socket(c) for c in cpus})
    print(f"{nombre:<34} T_mediana={med:7.3f}s  cpu/pared={ratio:.3f}  MHz_medio={statistics.fmean(mhz) if mhz else float('nan'):7.0f}  "
          f"cpus_usadas={sorted(cpus)} socket={socks}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--E", type=int, default=mlp.E_DEFAULT)
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    permitidas = sorted(os.sched_getaffinity(0))
    por_socket = {}
    for c in permitidas:
        por_socket.setdefault(_socket(c), []).append(c)
    print(f"# cpus permitidas={permitidas}", flush=True)
    print(f"# por socket={ {s: v for s, v in por_socket.items()} }", flush=True)
    sockets = sorted(por_socket)
    s_datos = sockets[0]
    os.sched_setaffinity(0, {por_socket[s_datos][0]})
    X, y, _, _ = mlp.cargar_datos(a.n)
    os.sched_setaffinity(0, set(permitidas))
    print(f"# datos cargados por el padre fijado en cpu {por_socket[s_datos][0]} (socket {s_datos}); n={a.n} E={a.E} B={a.B} reps={a.reps}", flush=True)

    medir("libre (sin fijar)", -1, False, X, y, a.E, a.B, a.reps)
    for s in sockets:
        cpu = por_socket[s][0]
        lugar = "mismo socket que los datos" if s == s_datos else "otro socket que los datos"
        medir(f"fijo cpu {cpu} ({lugar})", cpu, False, X, y, a.E, a.B, a.reps)
        medir(f"fijo cpu {cpu} + copia local datos", cpu, True, X, y, a.E, a.B, a.reps)
    medir("libre (sin fijar), repetición", -1, False, X, y, a.E, a.B, a.reps)


if __name__ == "__main__":
    main()

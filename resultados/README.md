# Resultados de Khipu

Todas las corridas usan el MLP 20-32-1 con E = 200 épocas, en un nodo de Khipu (Xeon Gold 6130, 2 sockets × 16 núcleos × 2 hilos). La cuenta de pregrado permite 32 CPUs de Slurm, que son **16 núcleos físicos**: con p = 32 cada núcleo ejecuta dos procesos por hyperthreading.

Cada configuración se repitió 3 veces y los resúmenes usan la mediana.

## Experimentos del proyecto

| Carpeta | Job | Nodo | Versión | Qué varía | Para qué sección |
|---|---|---|---|---|---|
| `01_beta3_casos_a_b_57130/` | 57130 | n004 | beta3 | **Caso A:** q = 32 fijo, p y n. **Caso B:** q = p, n | Impacto de los datos (n) y escalabilidad débil con q = p |
| `02_beta3_semillas_58166/` | 58166 | n005 | beta3 | n = 10 000 fijo, **q ∈ {32, 64, 128, 256} × p** | **Escalabilidad en semillas** (análisis principal pedido por el profesor) |
| `03_beta2_caso_a_58167/` | 58167 | n003 | beta2 | Caso A, igual que 01 | Comparación de versiones |
| `04_comparacion_versiones/` | — | — | beta2 vs beta3 | Generado con `python comparar_versiones.py` | Comparación de versiones |

**Versiones:**
- **beta1:** secuencial, con gradiente por bloques de 1024. Es la T_s de todos los speedups.
- **beta2:** paralela ingenua. Pool de procesos, gradiente con batch completo y Reduce lineal en el padre.
- **beta3:** paralela optimizada, el Algoritmo 2 del informe. Bloques de 1024, memoria compartida y Reduce en árbol.

## Archivos de cada corrida

- `resultados_*.csv`: crudo, una fila por ejecución (incluye las 3 repeticiones).
- `figuras/resumen_caso_*.csv`: mediana por configuración, con speedup y eficiencia ya calculados.
- `figuras/*.png`: figuras generadas por `analisis.py`.
- `figuras/ajuste.txt`: ajuste por mínimos cuadrados de `T_p = a·⌈q/p⌉·n + b·log2 p + c`.
- `slurm_*.out`: log de la ejecución, con la hora de cada medición.

**Columnas del CSV crudo:**

| Columna | Significado |
|---|---|
| `experimento` | `caso_a`, `caso_b` o `caso_q` |
| `version` | `beta1`, `beta2`, `beta3` o `beta3_pin` |
| `n`, `q`, `p` | muestras, semillas, procesos |
| `B` | tamaño de bloque del gradiente (0 = batch completo) |
| `rep` | repetición (0, 1, 2) |
| `t_total` | tiempo de pared total en s, con el arranque de procesos |
| `t_map`, `t_reduce` | tiempo de la fase Map y de la fase Reduce |
| `acc`, `seed` | accuracy y semilla del mejor modelo M* |
| `t_cpu_map` | tiempo de CPU sumado de los procesos en el Map |
| `cpu_ratio` | CPU / pared. ≈ 1 significa que el proceso nunca estuvo detenido |
| `nivcsw` | cambios de contexto involuntarios |
| `E` | **épocas** (¡no es la eficiencia!) |
| `h`, `d` | neuronas ocultas y entradas |
| `flops_modelo` | FLOPs para entrenar un modelo |

**Columnas extra de `resumen_caso_*.csv`:**

| Columna | Significado |
|---|---|
| `Ts` | tiempo secuencial con el mismo (n, q) |
| `S` | speedup T_s / T_p |
| `E` | **eficiencia** S / p |
| `flops_s` | FLOP/s |
| `T_teo` | tiempo que predice el ajuste |
| `t_std` | desviación estándar entre repeticiones |

## Hallazgos principales (punto de partida para el análisis)

1. **La eficiencia depende de p, no de q** (`02_.../figuras/caso_q_eficiencia_tabla.png`). Con n = 10 000, E ≈ 1.0, 0.91, 0.73, 0.65, 0.56 y 0.15 para p = 1, 2, 4, 8, 16 y 32, igual para q = 32 que para q = 256.
   - El modelo teórico E = 1 / (1 + p·log p / (q·n·C)) predice que E sube con q.
   - Que no suba indica que la pérdida no es el Reduce (log p, despreciable: `t_reduce` está en µs), sino una **ralentización de cada núcleo** cuando trabajan varios: ancho de banda de memoria compartido y frecuencia turbo que baja con más núcleos activos.
2. **Escalabilidad débil** (`caso_q_debil.png` y Caso B en `01_`). Con q/p constante, el tiempo sube con p en vez de quedarse plano. Es el mismo efecto anterior.
3. **beta2 se estanca y beta3 escala** (`04_.../comparacion_beta2_beta3.png`).
   - Con n = 40 000 y p = 16: beta2 tiene S = 2.2 y beta3 S = 7.0.
   - El batch completo de beta2 no cabe en caché, y todos los procesos compiten por la RAM.
   - La diferencia crece con n.
4. **p = 32 no son 32 núcleos reales.** Son hilos de hyperthreading: en n005, p = 32 tarda el doble que p = 16 (E = 0.15).
5. **El ajuste `T_p = a·⌈q/p⌉·n + b·log2 p + c` da R² alto pero constantes sin sentido físico** (b de segundos por nivel, c negativo). b absorbe la contención, no el costo del Reduce. Conviene un término de contención, por ejemplo T_p = a·⌈q/p⌉·n·(1 + γ·(p − 1)) + c, o explicarlo así en el informe.
6. **Empates:** con q = 256 hay dos semillas con la misma accuracy (54 y 192, acc = 0.8605). Según p, el árbol del Reduce elige una u otra. No es un error; conviene declararlo o fijar "en empate gana la semilla menor".

## Cuidados al comparar
- Las corridas 01, 02 y 03 se hicieron en **nodos distintos** (n004, n005, n003) del mismo modelo de CPU. Compara tendencias, no segundos exactos entre carpetas.
- En un nodo compartido, el mismo entrenamiento varía hasta ~1.6× según la frecuencia de la CPU (ver `diagnostico/numa_n004_57156`). Por eso se usan medianas.

## Diagnóstico (descarte de factores externos, no es resultado del proyecto)

| Carpeta | Qué se probó | Conclusión |
|---|---|---|
| `diagnostico/latencia_sleep_57124/` | Latencia al dormir procesos | Khipu no congela procesos (~60 µs por `sleep`) |
| `diagnostico/prueba_57128/` | Prueba de humo | El código corre en Khipu |
| `diagnostico/beta3_pin_57134/` | beta3 con cada proceso fijado a un núcleo | Eficiencia plana (~0.6), pero tiempos absolutos más lentos por la frecuencia del nodo |
| `diagnostico/numa_n003_57146/`, `numa_n004_57156/` | Memoria lejana (NUMA), pinning y MHz | NUMA y pinning descartados; el tiempo sigue a la frecuencia de la CPU |

## Reproducir figuras

```bash
python analisis.py --csv resultados/02_beta3_semillas_58166/resultados_semillas.csv --out resultados/02_beta3_semillas_58166/figuras
python comparar_versiones.py
```

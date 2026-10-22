# Entrenamiento paralelo de MLPs con semillas (CPD 2026-II, Proyecto 7)

Se entrenan q perceptrones multicapa con semillas distintas repartidas entre p procesos (Map) y se elige el de mayor accuracy con una reducción en árbol de log p niveles (Reduce). Modelo PRAM CREW implementado con `multiprocessing` en un solo nodo.

## Archivos

| Archivo | Rol |
|---|---|
| `mlp.py` | Datos sintéticos, MLP 20-h-1 en NumPy, entrenamiento por mini-batch acumulado |
| `_hilos.py` | Fija BLAS/OpenMP a 1 hilo por proceso |
| `beta1_secuencial.py` | Versión secuencial (T_s) |
| `beta2_pool.py` | Paralela ingenua: `ProcessPoolExecutor`, gradiente con batch completo (limitado por memoria), cada worker devuelve el modelo y el padre reduce en O(q) |
| `beta3_pram.py` | Paralela optimizada (algoritmo del informe): semillas `s mod p`, gradiente acumulado en bloques de 1024 que caben en caché, memoria compartida, Reduce en árbol de log p niveles con barreras (`--pin` fija cada proceso a un núcleo) |
| `bench.py` | Barrido de experimentos, escribe un CSV. beta1 y beta3 usan bloques `--B`; beta2 siempre batch completo |
| `analisis.py` | Speedup, eficiencia, ajuste `T_p = a·⌈q/p⌉·n + b·log2 p + c` y figuras |
| `numa_prueba.py` | Diagnóstico de NUMA y frecuencia de CPU |
| `slurm/` | Jobs de Khipu (`khipu.slurm` principal, `khipu_prueba.slurm` humo, `khipu_pin.slurm`, `khipu_beta2.slurm`) |
| `slurm/diagnostico/` | Jobs de diagnóstico de NUMA/frecuencia |

## Uso local

```bash
pip install -r requirements.txt
python beta3_pram.py --n 5000 --q 8 --p 4
python bench.py --ns 5000 --ps 1,2,4 --q 8 --reps 1 --out prueba.csv
python analisis.py --csv prueba.csv --out figs
```

## Uso en Khipu

Los scripts se lanzan **desde esta carpeta** (usan `$SLURM_SUBMIT_DIR`):

```bash
python3 -m venv ~/cpd-env && source ~/cpd-env/bin/activate && pip install -r requirements.txt
sbatch slurm/khipu_prueba.slurm
sbatch slurm/khipu.slurm
```

Requieren `module load gnu12/12.4.0 python3/3.13.2`. La cuenta de pregrado tiene un tope de 32 CPUs de Slurm (16 núcleos físicos con hyperthreading).

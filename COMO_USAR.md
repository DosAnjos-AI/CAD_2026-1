# Como Usar

## Pré-requisitos

- `gcc` com suporte a OpenMP (`-fopenmp`)
- CUDA Toolkit compatível com a GPU local
- `nvidia-smi` (GPUs discretas) ou `tegrastats` (Jetson AGX Orin)
- `make`

## Clonar e executar

```bash
git clone https://github.com/DosAnjos-AI/CAD_2026-1
cd CAD_2026-1
```

**Sem sudo** (energia CPU = N/A):
```bash
bash scripts/run_benchmarks.sh
```

**Com sudo** (coleta energia CPU via `perf stat`, requer configuração sudoers):
```bash
sudo bash scripts/run_benchmarks_sudo.sh
```

Parâmetros padrão: HARDWARE=mx350, ITERACOES=10, RUNS=1500.
Execuções nunca são interrompidas por timeout.

## Estrutura de uma execução

Cada binário executa **1 rodada de aquecimento** (verifica corretude, descarta o tempo),
depois mede `RUNS` execuções consecutivas. `tempo_total_s` é o tempo total; médio: `tempo_total_s / runs`.

## Resultados

```
results/<hardware>/
  <algoritmo>_<api>.csv   — dados de cada iteração
  benchmark_log.txt       — log completo de todas as execuções
```

Formato CSV (9 campos):
```
algoritmo,api,hardware,tamanho,iteracao,tempo_total_s,energia_gpu_j,energia_cpu_j,corretude
mergesort,openmp,mx350,100000,1,0.514804,0.0000,4.29,OK
```

## Estimativa de tempo na MX350

Benchmark completo: **~46–50 horas** (RUNS=1500, ITERACOES=10).
Caso mais custoso: Mergesort CUDA DP com N=100.000 (~33h nas 10 iterações).
Hardwares mais potentes (ex: RTX 4090) serão significativamente mais rápidos.

## Configuração para coleta de energia CPU

O `run_benchmarks_sudo.sh` usa `sudo perf stat -e power/energy-pkg/`. Para rodar
em background sem prompt de senha, configure **uma vez em cada máquina**:

```bash
sudo visudo
# Adicionar: SEU_USUARIO ALL=(ALL) NOPASSWD: /usr/bin/perf
```

Verificar: `sudo -n perf stat -e power/energy-pkg/ sleep 1`
Sem essa configuração, use `run_benchmarks.sh` — `energia_cpu_j` ficará como `N/A`.

## Diferença entre os dois scripts

| Aspecto | `run_benchmarks.sh` | `run_benchmarks_sudo.sh` |
|---------|--------------------|-----------------------|
| Requer sudo | Não | Sim |
| Energia CPU | N/A | `perf stat -e power/energy-pkg/` |
| Energia GPU | nvidia-smi | idem |

## Observações

- **Execuções rápidas (< 1s):** energia GPU pode ser N/A — `nvidia-smi` amostra a cada 1s.
- **Interrupção:** `Ctrl+C` encerra o script; CSVs gerados até o momento são mantidos.
- **Teste rápido:** `TEST_MODE=1 RUNS=5 bash scripts/run_benchmarks.sh` (mergesort/openmp/N=100, 1 iteração).

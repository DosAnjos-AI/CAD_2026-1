# CAD_2026-1 — Análise de Desempenho e Consumo Energético de Aplicações Recursivas

Replicação e extensão de **Nogueira et al. (SSCAD 2024)**:
> "Análise de Desempenho e Consumo Energético de Aplicações Recursivas em Ambientes OpenMP, CUDA e CUDA DP"

Objetivo: validar os resultados originais do artigo e adicionar variantes otimizadas de cada algoritmo, medindo tempo de execução e consumo energético em três plataformas de hardware distintas.

---

## Estrutura de Pastas

```
CAD_2026-1/
├── mergesort/
│   ├── artigo/        # implementacao fiel ao artigo
│   ├── otimizado/     # implementacao otimizada
│   └── sequencial/    # baseline sequencial
├── quicksort/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── bfs/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── sssp/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── results/
│   ├── benchmark.csv  # dados coletados
│   └── benchmark.log  # log de execucao
├── run_all.sh         # orquestrador principal
├── README.md
├── HOWTO.md
└── REQUIREMENTS.md
```

---

## Hardware Suportado

| Maquina | GPU | CPU | Arquitetura CUDA |
|---|---|---|---|
| Notebook (MX350) | NVIDIA GeForce MX350 | Intel i7-1165G7 | sm_61 |
| Servidor (RTX 4090) | NVIDIA GeForce RTX 4090 | AMD Ryzen 9 3900X | sm_89 |
| Jetson AGX Orin | NVIDIA Ampere (integrada) | ARM Cortex-A78AE | sm_87 |

---

## Algoritmos e APIs

- **Algoritmos**: Mergesort, Quicksort, BFS, SSSP
- **APIs**: OpenMP, CUDA, CUDA DP (Dynamic Parallelism)
- **Variantes por algoritmo**: artigo (fiel ao original), otimizado, sequencial (baseline)
- **Total de scripts**: 28

---

## Metodologia de Execucao

| Parametro | Valor |
|---|---|
| Iteracoes por configuracao | 25 |
| Execucoes por iteracao | 400 |
| Total de execucoes por configuracao | 10.000 |
| Warmup | 1 iteracao descartada antes das 25 medidas |
| Sleep entre tamanhos | 30 segundos (isolamento termico) |
| Sleep entre binarios | 20 segundos |

Ordem de execucao fixa: mergesort → quicksort → bfs → sssp, openmp → cuda → cuda_dp, artigo → otimizado → sequencial.

---

## Tamanhos de Entrada por Algoritmo

| Algoritmo | Tamanhos |
|---|---|
| Mergesort | 100, 10.000, 100.000 elementos |
| Quicksort | 100, 10.000, 100.000 elementos |
| BFS | 10K×30K, 100K×300K, 500K×1M (nos×arestas) |
| SSSP | 1K×4K, 10K×30K, 100K×300K, 200K×400K (nos×arestas) |

---

## Formato do CSV de Saida

```
algoritmo|api|versao|hardware|tamanho|iteracao|tempo_total_s|energia_gpu_j|energia_cpu_j|corretude
```

| Campo | Descricao |
|---|---|
| algoritmo | mergesort, quicksort, bfs, sssp |
| api | openmp, cuda, cuda_dp, sequencial |
| versao | artigo, otimizado, sequencial |
| hardware | mx350, rtx4090, jetson |
| tamanho | numero de elementos ou nos |
| iteracao | 1 a 25 |
| tempo_total_s | tempo medio por execucao (segundos, decimal virgula) |
| energia_gpu_j | energia GPU em joules (NA se indisponivel) |
| energia_cpu_j | energia CPU em joules via RAPL/perf (NA se indisponivel) |
| corretude | OK ou FAIL |

Separador: `|` — Decimal: `,` (virgula) — Append, nunca sobrescreve linhas existentes.

---

## Coleta de Energia

- **CPU**: Intel RAPL via `perf stat -e power/energy-pkg/` (primario) ou `/sys/class/powercap/intel-rapl:0/energy_uj` (fallback)
- **GPU RTX 4090**: `nvidia-smi dmon -s p -d 100`
- **GPU Jetson**: `tegrastats --interval 100`
- **GPU MX350**: sem sensor INA — registrado como `NA`

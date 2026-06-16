# CLAUDE.md — Projeto CAD_2026_v3

## Contexto

Projeto de benchmarking de algoritmos paralelos para a disciplina de Computação de Alto Desempenho (CAD 2026-1).
Objetivo: comparar speedup relativo entre APIs (CPU sequencial, OpenMP, CUDA, CUDA DP) para 4 algoritmos.
Duas branches paralelas para replicação:
- `CAD_2026_v3` — ordem: bitonic_sort → merge_sort → bfs → floyd_warshall
- `CAD_2026_v3_inv` — ordem: floyd_warshall → bfs → merge_sort → bitonic_sort

## Regras obrigatórias de código

- Comentários obrigatoriamente em português
- Sem emojis em código (prints, logs, comentários)
- Sem código morto (trechos comentados, variáveis não usadas, includes desnecessários)
- Filosofia KISS: solução mais simples que resolve o problema corretamente
- Sem over-engineering ou abstrações desnecessárias
- Todo script deve ser testado e validado antes de considerar concluído

## Linguagens por API

- CPU sequencial: `.c` compilado com `gcc`
- OpenMP: `.c` compilado com `gcc -fopenmp`
- CUDA: `.cu` compilado com `nvcc`
- CUDA DP: `.cu` compilado com `nvcc -rdc=true`

## Flags de compilação

```bash
# CPU e OpenMP
gcc -O3 -march=native -fopenmp -o binario arquivo.c

# CUDA
nvcc -O3 -arch=sm_XX -o binario arquivo.cu

# CUDA DP
nvcc -O3 -arch=sm_XX -rdc=true -o binario arquivo.cu
```

`sm_XX` detectado dinamicamente via `nvidia-smi` no `run_all.sh`.
`NUM_THREADS` detectado via `nproc` no `run_all.sh` e passado como `-DNUM_THREADS=$(nproc)`.

## Estrutura de diretórios

```
CAD_2026-1/
├── run_all.sh
├── CLAUDE.md
├── results/
│   └── resultados.csv
├── bitonic_sort/
│   ├── bitonic_cpu.c
│   ├── bitonic_openmp.c
│   ├── bitonic_cuda.cu
│   └── bitonic_cudadp.cu
├── merge_sort/
│   ├── merge_cpu.c
│   ├── merge_openmp.c
│   ├── merge_cuda.cu
│   └── merge_cudadp.cu
├── bfs/
│   ├── bfs_cpu.c
│   ├── bfs_openmp.c
│   ├── bfs_cuda.cu
│   └── bfs_cudadp.cu
└── floyd_warshall/
    ├── floyd_cpu.c
    ├── floyd_openmp.c
    ├── floyd_cuda.cu
    └── floyd_cudadp.cu
```

Binários compilados ficam dentro do diretório do algoritmo correspondente.

## Formato do CSV

Arquivo: `results/resultados.csv`

```
sep=|
algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos
```

### Regras do CSV

- Separador: pipe `|`
- Decimais: vírgula `,` (locale brasileiro)
- Primeira linha: `sep=|` (compatibilidade Excel)
- Segunda linha: cabeçalho
- Política: append-only, nunca sobrescrever linhas existentes
- `tempo_s`: média aritmética das execuções internas (float com vírgula)
- `corretude`: `1` se resultado correto, `0` se incorreto
- `threads_blocos`: configuração usada (ex: `256x4096` para CUDA, `32x1` para OpenMP, `1x1` para CPU)
- `cenario`: sempre `aleatorio` neste projeto
- `tamanho`: valor de N (vetores) ou V (grafos) como inteiro

### Exemplo de linhas

```
bitonic_sort|cpu|aleatorio|1048576|1|0,823|1|1x1
bitonic_sort|cpu|aleatorio|1048576|2|0,819|1|1x1
bitonic_sort|openmp|aleatorio|1048576|1|0,312|1|32x1
bitonic_sort|cuda|aleatorio|1048576|1|0,003|1|256x4096
```

## Protocolo de execução por combinação

Para cada combinação `algoritmo|api|tamanho`:

1. Verificar se já existem 10 linhas completas no CSV para essa combinação
2. Se completa: pular inteiramente (incluindo warmup)
3. Se incompleta: apagar linhas parciais dessa combinação e reprocessar do zero
4. Se ausente: executar normalmente

### Sequência de execução

```
3 execuções de warmup  → descartadas, sem registro no CSV
10 iterações medidas   → cada iteração executa o algoritmo 10 vezes internamente
                         tempo_s = média das 10 execuções internas
                         cada iteração gera 1 linha no CSV
```

Total por combinação: 130 execuções (3×10 warmup + 10×10 medidas)

## Tipo de dado

`int32` em todos os algoritmos e APIs.

## Validação de corretude

Comparação elemento a elemento contra a versão sequencial CPU (oráculo).
Comparação exata (sem epsilon) — garantida pelo uso de `int32`.

- Vetores (Bitonic/Merge): vetor de saída idêntico ao `qsort` da libc
- BFS: array de distâncias idêntico ao BFS sequencial
- Floyd-Warshall: matriz de distâncias idêntica à versão sequencial

## Tamanhos de entrada

### Vetores (Bitonic Sort e Merge Sort) — potências de 2 obrigatório

```
2^10  =      1.024
2^12  =      4.096
2^14  =     16.384
2^16  =     65.536
2^18  =    262.144
2^20  =  1.048.576
2^22  =  4.194.304
2^24  = 16.777.216
2^26  = 67.108.864
```

### BFS — grafo aleatório Erdős–Rényi, grau médio 16

```
V = 2^10  =      1.024   E ≈     16.384
V = 2^12  =      4.096   E ≈     65.536
V = 2^14  =     16.384   E ≈    262.144
V = 2^16  =     65.536   E ≈  1.048.576
V = 2^18  =    262.144   E ≈  4.194.304
V = 2^20  =  1.048.576   E ≈ 16.777.216
V = 2^22  =  4.194.304   E ≈ 67.108.864
V = 2^24  = 16.777.216   E ≈ 268.435.456
```

Representação: CSR (Compressed Sparse Row)
Fonte vértice: sempre vértice 0

### Floyd-Warshall — matriz de adjacência densa, pesos int32 aleatórios [1, 1000]

```
V = 2^5  =     32
V = 2^6  =     64
V = 2^7  =    128
V = 2^8  =    256
V = 2^9  =    512
V = 2^10 =  1.024
V = 2^11 =  2.048
V = 2^12 =  4.096
V = 2^13 =  8.192
```

INF = 1e9 (sentinela para arestas inexistentes)
Diagonal principal = 0

## Ordem de execução no run_all.sh

### CAD_2026_v3 (Máquina A)
```
bitonic_sort  → merge_sort → bfs → floyd_warshall
```

### CAD_2026_v3_inv (Máquina B)
```
floyd_warshall → bfs → merge_sort → bitonic_sort
```

Dentro de cada algoritmo (ambas as branches):
```
cpu → openmp → cuda → cudadp
tamanhos: crescente (menor para maior)
```

Sleep de 3 segundos entre algoritmos distintos para isolamento térmico.
Sem sleep entre execuções individuais dentro do mesmo algoritmo.

## Comportamento do run_all.sh

- Sem argumentos: chamado apenas como `./run_all.sh`
- Detecta SM target via `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
- Detecta threads via `nproc`
- Compila todos os binários antes de iniciar execuções
- Verifica existência do binário antes de cada execução
- Cria `results/` e `results/resultados.csv` se não existirem
- Execução via: `nohup ./run_all.sh > results/log.txt 2>&1 &`

## Hardware alvo

- CPU: Intel Core i9-14900K — 24 núcleos / 32 threads / 6.0 GHz
- GPU: NVIDIA GeForce RTX 4090 — 24GB VRAM / sm_89
- OS: Ubuntu 24.04.3 LTS
- CUDA: 12.1.105
- Compilador: gcc-12 / nvcc inline
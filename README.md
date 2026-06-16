# CAD_2026-1 — Benchmark de Algoritmos Paralelos

Projeto da disciplina de Computação de Alto Desempenho (CAD 2026-1).

## 1. Objetivo

Comparar o speedup relativo entre quatro APIs de paralelização — CPU
sequencial, OpenMP, CUDA e CUDA Dynamic Parallelism (CUDA DP) — aplicadas a
quatro algoritmos clássicos com perfis de paralelismo distintos: dois de
ordenação (Bitonic Sort, Merge Sort) e dois sobre grafos (BFS, Floyd-Warshall).

O projeto replica a metodologia em duas máquinas com ordens de execução
invertidas (branches `CAD_2026_v3` e `CAD_2026_v3_inv`), de forma a isolar
efeitos de ordem/aquecimento térmico do hardware sobre os tempos medidos.

**Hardware alvo**: Intel Core i9-14900K (24 núcleos / 32 threads, 6.0 GHz) +
NVIDIA GeForce RTX 4090 (24 GB VRAM, `sm_89`), Ubuntu 24.04.3 LTS, CUDA 12.1.105.

## 2. Algoritmos implementados

| Algoritmo | CPU | OpenMP | CUDA | CUDA DP |
|---|---|---|---|---|
| **Bitonic Sort** | Iterativo, 3 loops (`k`, `j`, `i`), `compare_and_swap` inline | Paraleliza o loop `i` com `parallel for`; barrier implícito entre passos `(k,j)` | Um kernel por passo `(k,j)`, `N/2` threads via mapeamento sem checagem condicional (`i = (tid/j)*(j*2) + (tid%j)`) | Kernel pai `<<<1,1>>>` controla o loop `(k,j)` no device e lança o mesmo kernel filho do CUDA flat |
| **Merge Sort** | Recursivo top-down, `merge` iterativo, buffer único reaproveitado | `#pragma omp task` por metade com cutoff em 1024 elementos | Bottom-up iterativo, kernel por passo, merge via *co-rank* (merge path) para ocupação plena da GPU | Kernel pai se auto-relança nas duas metades e usa `cudaStreamTailLaunch` para o merge final |
| **BFS** | Por nível, fronteiras ping-pong pré-alocadas de tamanho V | Fronteira paralelizada com buffer local por thread (1024) e CAS para evitar duplicidade | Um thread por vértice, `atomicCAS` para reivindicar `dist[]`, fronteira via flags | Kernel pai por nível lança kernel filho só para vértices ativos, expandindo seus vizinhos |
| **Floyd-Warshall** | Triplo loop `k→i→j`, poda para evitar overflow ao somar `INF` | Loop `i` paralelizado (`collapse` não usado — `private(j)`), `k` sequencial | Kernel 2D por `(i,j)`, lançado V vezes (um por `k`), blocos 16×16 | Kernel pai relança a si mesmo via `cudaStreamTailLaunch` a cada `k`, mesmo kernel de atualização do CUDA flat |

Detalhes completos de cada implementação, incluindo decisões técnicas
específicas e desvios em relação ao planejamento original, estão nos
`README.md` de cada diretório (`bitonic_sort/`, `merge_sort/`, `bfs/`,
`floyd_warshall/`).

## 3. Metodologia

- **Tipo de dado**: `int32_t` em todas as implementações e APIs.
- **Geração de dados**: determinística via `srand(42)` — mesma semente em
  todas as APIs, garantindo a mesma entrada (vetor ou grafo) entre execuções
  e implementações.
- **Cenário**: sempre `aleatorio`.
- **Protocolo de execução por combinação** `algoritmo|api|tamanho`:
  - 3 execuções de warmup, descartadas (sem registro no CSV)
  - 10 iterações medidas, cada uma com 10 execuções internas
  - `tempo_s` = média aritmética das 10 execuções internas da iteração
  - Total: 130 execuções por combinação
- **Métrica de comparação**: speedup relativo entre APIs, calculado a partir
  do `tempo_s` médio de cada combinação.
- **Validação de corretude**: comparação exata (sem epsilon, garantida pelo
  uso de `int32_t`) contra o oráculo sequencial — `qsort` da libc para os
  vetores, BFS sequencial para distâncias, versão sequencial para a matriz
  do Floyd-Warshall.

### Tamanhos testados

| Categoria | Algoritmos | Valores |
|---|---|---|
| Vetores (potências de 2) | Bitonic Sort, Merge Sort | 1.024 · 4.096 · 16.384 · 65.536 · 262.144 · 1.048.576 · 4.194.304 · 16.777.216 · 67.108.864 |
| Grafos Erdős–Rényi, grau médio 16 | BFS | V = 1.024 · 4.096 · 16.384 · 65.536 · 262.144 · 1.048.576 · 4.194.304 · 16.777.216 |
| Matriz densa de adjacência | Floyd-Warshall | V = 32 · 64 · 128 · 256 · 512 · 1.024 · 2.048 · 4.096 · 8.192 |

## 4. Estrutura do repositório

```
.
├── run_all.sh              orquestrador: compila e executa todo o benchmark
├── CLAUDE.md                regras e definicoes do projeto
├── decisoes_tecnicas.md     decisoes de implementacao por algoritmo
├── HOWTO.md                 guia pratico de uso
├── README.md                este arquivo
├── results/
│   ├── resultados.csv       CSV append-only com todos os resultados
│   └── log.txt              log gerado por nohup ao rodar run_all.sh
├── bitonic_sort/
│   ├── README.md
│   ├── bitonic_cpu.c / bitonic_openmp.c / bitonic_cuda.cu / bitonic_cudadp.cu
│   └── binarios compilados (mesmo nome, sem extensao)
├── merge_sort/         (mesma estrutura, prefixo merge_)
├── bfs/                 (mesma estrutura, prefixo bfs_)
└── floyd_warshall/      (mesma estrutura, prefixo floyd_)
```

## 5. Como executar

Benchmark completo, em background, sobrevivendo a desconexão de terminal:

```bash
nohup ./run_all.sh > results/log.txt 2>&1 &
```

O script detecta automaticamente a arquitetura da GPU (`nvidia-smi`) e o
número de threads da CPU (`nproc`), compila todos os binários e processa
cada combinação `algoritmo|api|tamanho`, pulando as que já estão completas
no CSV. Para execução de um algoritmo isolado, retomada após interrupção e
solução de problemas comuns, ver [`HOWTO.md`](HOWTO.md).

## 6. Formato de saída

Arquivo: `results/resultados.csv`, separador `|`, decimais com vírgula
(locale brasileiro), primeira linha `sep=|` para compatibilidade com Excel.

```
sep=|
algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos
```

| Campo | Descrição |
|---|---|
| `algoritmo` | `bitonic_sort`, `merge_sort`, `bfs` ou `floyd_warshall` |
| `api` | `cpu`, `openmp`, `cuda` ou `cudadp` |
| `cenario` | sempre `aleatorio` |
| `tamanho` | N (vetores/BFS) ou V (Floyd-Warshall), inteiro |
| `iteracao` | número da iteração medida (1 a 10) |
| `tempo_s` | tempo médio das 10 execuções internas da iteração |
| `corretude` | `1` se idêntico ao oráculo sequencial, `0` caso contrário |
| `threads_blocos` | configuração de paralelismo usada na execução |

Para comparar APIs em uma mesma combinação `algoritmo|tamanho`, calcula-se o
speedup como a razão entre o `tempo_s` médio da CPU sequencial e o `tempo_s`
médio da API em questão. Detalhes de interpretação e exemplos de linhas
reais estão em [`HOWTO.md`](HOWTO.md#5-formato-do-csv-de-saída).

## 7. Decisões técnicas relevantes

- **`int32_t` em todos os algoritmos e APIs**: garante comparação de
  corretude exata, sem epsilon, entre CPU e GPU — não há erro de
  arredondamento de ponto flutuante a considerar.
- **Merge Sort CUDA em bottom-up, não top-down**: a versão CUDA flat evita
  recursão no device (custosa e com profundidade de pilha incerta) e usa
  *co-rank* (merge path) para que cada thread compute independentemente sua
  posição de saída — sem isso, os últimos passos do bottom-up (poucos pares
  grandes) deixariam a maior parte da GPU ociosa.
- **`cudaStreamTailLaunch` em vez de `cudaDeviceSynchronize()` no device**:
  usado em Merge Sort DP e Floyd-Warshall DP para encadear estágios
  dependentes (merge final, iteração `k+1`) sem sincronização explícita do
  device — mais simples e mais barato sob CDP2 (CUDA 12.1, modelo único a
  partir do CUDA 12.0; `cudaDeviceSynchronize()` chamado pelo device está
  depreciado nesse modelo).
- **Nota honesta sobre CUDA DP em algoritmos regulares**: Bitonic Sort,
  Merge Sort e Floyd-Warshall têm estrutura de paralelismo estática e
  balanceada — o benefício típico do CUDA DP (adaptar paralelismo a cargas
  irregulares) não se aplica bem a eles, e o overhead de lançamento de
  kernel a partir do device tende a não ser compensado. Espera-se CUDA DP
  igual ou mais lento que CUDA flat nesses três casos. A exceção é o BFS,
  onde a irregularidade da fronteira (vértices ativos variam por nível)
  justifica de forma mais natural o uso de paralelismo dinâmico — ainda que,
  para o grafo Erdős–Rényi de grau uniforme usado aqui, a expectativa também
  seja de desempenho inferior ao CUDA flat. Em todos os casos, um resultado
  de CUDA DP mais lento é tratado como achado válido do benchmark, não como
  falha de implementação.

## 8. Pendências conhecidas

- **Validação funcional na RTX 4090**: os binários CUDA e CUDA DP dos quatro
  algoritmos foram compilados e exercitados localmente (incluindo em GPU
  `sm_61`, distinta do hardware alvo), mas a execução de medição e validação
  de corretude no hardware alvo do projeto (RTX 4090, `sm_89`) ainda não foi
  realizada.
- **Branch `CAD_2026_v3_inv`**: replicação da Máquina B, com ordem de
  execução invertida dos algoritmos (`floyd_warshall → bfs → merge_sort →
  bitonic_sort`), ainda não executada — destinada a isolar efeitos de
  ordem/aquecimento térmico sobre os tempos medidos.

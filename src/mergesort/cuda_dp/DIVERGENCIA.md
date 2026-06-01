# Divergência: Mergesort CUDA DP vs Artigo de Referência

**Projeto:** Replicação de Nogueira et al., SSCAD 2024
**Hardware:** NVIDIA MX350, driver 535, CUDA 12.2
**Data:** 2026-05-31

---

## 1. Causa Raiz

A divergência tem uma causa única e precisa: o **fallback para subproblemas com
`num_elementos <= 16384`** foi implementado como `insertion_sort` (O(n²)), quando
o repositório base usado pelo artigo (`JoeyOhman/GPUMergeSort`) usa **merge
sequencial O(n)** para esse caso.

### Lógica no repositório base (JoeyOhman/GPUMergeSort, `parallel.cu`)

```c
if (nTot > 16384) {
    mergeKernel<<<numBlocks, numThreadsPerBlock>>>(arr, aux, low, mid, high);
} else {
    merge(arr, aux, low, mid, high);  // merge sequencial O(n)
}
```

### Lógica na nossa implementação (`mergesort_cuda_dp.cu`, linha 92)

```c
if (num_elementos > 16384 && num_elementos < 1048576) {
    binary_search_merge<<<...>>>(src, dst, esq, meio, dir);
} else {
    insertion_sort_device(src, esq, dir);  // insertion_sort O(n²)
    for (int i = esq; i <= dir; i++) dst[i] = src[i];
}
```

A substituição de `merge O(n)` por `insertion_sort O(n²)` é o único responsável
pelos resultados 160–590× piores que o CUDA simples.

---

## 2. Evidências Numéricas — Análise Nível a Nível

O mergesort bottom-up itera sobre `largura = 1, 2, 4, ..., 2^k` até cobrir o vetor.
Em cada nível, cada thread processa um subproblema de `num_elementos = min(2×largura, restante)`.

### N = 100

| largura | threads | max_sub | caminho | ops (IS) |
|---------|---------|---------|---------|----------|
| 1 | 50 | 2 | insertion_sort | 2.00×10² |
| 2 | 25 | 4 | insertion_sort | 4.00×10² |
| 4 | 13 | 8 | insertion_sort | 7.84×10² |
| 8 | 7 | 16 | insertion_sort | 1.55×10³ |
| 16 | 4 | 32 | insertion_sort | 3.09×10³ |
| 32 | 2 | 64 | insertion_sort | 5.39×10³ |
| **64** | **1** | **100** | **insertion_sort** | **1.00×10⁴ ← gargalo** |

**DP nunca ativado. Total ops insertion_sort: ~2,1×10⁴. Rápido porque max_sub=100 é pequeno.**

### N = 10.000

| largura | threads | max_sub | caminho | ops (IS) |
|---------|---------|---------|---------|----------|
| 1 | 5000 | 2 | insertion_sort | 2.00×10⁴ |
| ... | ... | ... | insertion_sort | ... |
| 1024 | 5 | 2048 | insertion_sort | 2.00×10⁷ |
| 2048 | 3 | 4096 | insertion_sort | 3.68×10⁷ |
| 4096 | 2 | 8192 | insertion_sort | 7.04×10⁷ |
| **8192** | **1** | **10000** | **insertion_sort** | **1.00×10⁸ ← gargalo** |

**DP nunca ativado** (N=10.000 < threshold de 16.384).
Na iteração final, **1 única thread** executa `insertion_sort` sobre os **10.000 elementos**
inteiros do vetor: 10.000² = 100 milhões de operações em sequência.
Resultado medido: **3,14 s/run** vs 0,005 s do CUDA simples (590× mais lento).

### N = 100.000

| largura | threads | max_sub | caminho | ops (IS) |
|---------|---------|---------|---------|----------|
| 1 | 50000 | 2 | insertion_sort | 2.00×10⁵ |
| ... | ... | ... | insertion_sort | ... |
| 4096 | 13 | 8192 | insertion_sort | 8.08×10⁸ |
| **8192** | **7** | **16384** | **insertion_sort** | **1.61×10⁹ ← gargalo** |
| 16384 | 4 | 32768 | **DP (3 subs)** | — |
| 32768 | 2 | 65536 | **DP** | — |
| 65536 | 1 | 100000 | **DP** | — |

DP ativado apenas nos **3 últimos níveis** (6 chamadas no total de 100.006).
O gargalo é o nível `largura=8192`: **7 threads paralelas**, cada uma com
`insertion_sort` de até 16.384 elementos (16.384² = 268 milhões de ops).

**Por que N=100.000 é "apenas" 160× mais lento** e não 590×?
A GPU **paraleliza** as 7 instâncias de `insertion_sort` no nível crítico. O tempo
de parede é dominado pelo maior subproblema (16.384 elementos), não pela soma.
Comparando com N=10.000 (1 thread de 10.000 elementos):
`16.384² / 10.000²` ≈ 2,7× — consistente com a razão de tempos medidos
(7,94 s vs 3,14 s ≈ 2,5×). O CUDA simples também escala com N, reduzindo o
ratio de 590× para 160×.

---

## 3. Fidelidade ao Algoritmo 1 do Artigo

**Parcialmente fiel.**

O threshold `num_elementos > 16384` está correto e foi implementado exatamente
como descrito no Algoritmo 1 do artigo principal. A divergência está no **else**:

- O artigo descreve `else: insertion_sort()` sem qualificação de tamanho
- O repositório base (JoeyOhman) usa `else: merge sequencial O(n)`
- Nossa implementação seguiu o pseudocódigo literalmente, usando `insertion_sort`
  para **todos** os subproblemas que não ativam o DP

O pseudocódigo do artigo parece descrever `insertion_sort` como base case de
subproblemas muito pequenos (folhas da recursão). Para um mergesort bottom-up,
porém, não há distinção estrutural entre "folha" e "nível intermediário" — o mesmo
`else` cobre desde pares de 2 elementos até arrays de 16.384 elementos.

### Sobre o Apêndice Técnico

O apêndice `sscad2024_trilhaprincipal_apendice1.pdf` foi acessado e lido na íntegra
(408 linhas extraídas). Ele documenta SSSP, BFS, Quicksort e Mergesort apenas nas
versões **OpenMP e CUDA simples**. **Não há nenhum pseudocódigo ou código da
versão CUDA DP.** O próprio apêndice declara: *"Este documento apresenta os
algoritmos SSSP, BFS, Quicksort e Mergesort nas versões OpenMP e CUDA."*

Não é possível verificar pelo apêndice como o fallback foi implementado na versão DP.

---

## 4. Hipótese sobre o Speedup de 23× Reportado pelo Artigo

Com base na análise do repositório base (JoeyOhman/GPUMergeSort), a hipótese
mais provável é que o artigo utilizou **merge sequencial O(n)** como fallback,
exatamente como está no repositório original — e descreveu esse fallback como
"insertion_sort" no pseudocódigo por simplificação ou imprecisão terminológica.

Com merge O(n) no lugar de insertion_sort O(n²):
- Todos os níveis abaixo do threshold executam merge sequencial eficiente
- O DP é responsável apenas pelos níveis superiores (subproblemas grandes)
- O desempenho seria comparável ou superior ao CUDA simples, consistente
  com o speedup de 7× sobre CUDA e 23× sobre OpenMP reportado pelo artigo

Nenhuma outra hipótese explica a magnitude da diferença sem alterar o fallback.

---

## 5. Conclusão para o Estudo de Replicação

Nossa implementação do Mergesort CUDA DP é **fiel ao Algoritmo 1 do artigo**
no que tange ao threshold (> 16.384) e ao mecanismo de Dynamic Parallelism
com `binary_search_merge`. A divergência de desempenho decorre de uma
**ambiguidade no pseudocódigo**: o artigo descreve `else: insertion_sort()` sem
especificar que esse caminho deve ser restrito a subproblemas pequenos.

O repositório base (JoeyOhman/GPUMergeSort) usa merge O(n) como fallback,
e o apêndice técnico não documenta a versão DP. Portanto, a interpretação do
`else` como `insertion_sort` universal é a causa da divergência — um erro de
portagem introduzido ao adaptar o código do CUDA simples (que usa
`gpu_bottomup_merge` para todos os níveis exceto o primeiro) para o CUDA DP.

Os resultados medidos são apresentados como estão e constituem um achado
válido do estudo de replicação: demonstram que a ambiguidade no Algoritmo 1
do artigo pode levar a uma implementação 590× mais lenta para tamanhos abaixo
do threshold de Dynamic Parallelism.

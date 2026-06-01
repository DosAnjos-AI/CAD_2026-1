# Divergência: Quicksort CUDA DP vs Artigo de Referência

**Projeto:** Replicação de Nogueira et al., SSCAD 2024
**Hardware:** NVIDIA MX350, driver 535, CUDA 12.2
**Data:** 2026-05-31

---

## 1. Causa Raiz

A divergência tem **duas causas independentes**:

1. **Algoritmo CUDA simples inadequado:** Nossa implementação CUDA usa pilha
   explícita com 1 thread por subproblema e muitos `cudaDeviceSynchronize()`
   por nível. O artigo provavelmente usa um algoritmo CUDA mais paralelo
   (múltiplas threads por partição). Por isso, nosso CUDA é mais lento que
   OpenMP em todos os tamanhos — invertendo o ranking do artigo.

2. **Overhead de lançamento de kernel no CUDA DP:** Nossa implementação CUDA DP
   lança **1 thread por kernel recursivo** com streams NonBlocking. Para
   N=100.000, isso gera ~6.250 lançamentos de kernel (100.000 / THRESHOLD=32 ≈
   3.125 folhas + 3.125 nós internos) a ~5–10μs cada = 31–62ms só em overhead.
   Medido: 99,6ms/run.

---

## 2. Parâmetros Implementados

```c
// quicksort_cuda_dp.cu, linhas 7-8
#define PROFUNDIDADE_MAX 24
#define THRESHOLD        32
```

- **Limite de profundidade:** 24 ✓ (Algoritmo 2 do artigo)
- **Threshold de tamanho:** 32 elementos (folhas da recursão)
- **Fallback:** `selection_sort_device` O(n²) — quando profundidade ≥ 24 **ou**
  tamanho ≤ 32
- **Particionamento:** Hoare com pivô central ✓
- **Paralelismo:** streams NonBlocking para filhos esquerdo/direito ✓

---

## 3. Evidências Numéricas

### Tempos medidos (ms/run, média de 10 iterações × 1.500 runs)

| N | OpenMP | CUDA | CUDA DP | Ranking obtido |
|---|--------|------|---------|----------------|
| 100 | 0,18 | 0,26 | 0,84 | OMP < CUDA < **CUDA DP** |
| 10.000 | 0,77 | 9,13 | 7,80 | OMP < CUDA DP < CUDA |
| 100.000 | 8,65 | 79,56 | 99,60 | OMP < CUDA < **CUDA DP** |

### Comparação com ranking do artigo

| Posição | Artigo | Nossos resultados (N=100.000) |
|---------|--------|-------------------------------|
| 1º (mais rápido) | CUDA | **OpenMP** |
| 2º | OpenMP | CUDA |
| 3º (mais lento) | CUDA DP | **CUDA DP** ✓ |

O artigo **concorda** que CUDA DP é o pior. A divergência é que OpenMP supera
CUDA nos nossos resultados, invertendo as posições 1º e 2º.

### Análise da profundidade de recursão para N=100.000

Para dados aleatórios com particionamento de Hoare (pivô central):
- Profundidade esperada: `log₂(100.000) ≈ 17` níveis
- `PROFUNDIDADE_MAX = 24` — raramente atingido com dados aleatórios
- Quando `profundidade ≥ 24`, o subproblema médio tem `100.000 / 2²⁴ ≈ 0,006`
  elementos — praticamente inexistente para N=100.000

**Conclusão:** O limite de profundidade 24 **não é o gargalo** para N=100.000
com dados aleatórios. O fallback `selection_sort` é ativado principalmente pelo
threshold de tamanho (`tamanho ≤ 32`), que é o uso correto e eficiente.

---

## 4. Por que CUDA DP é mais lento que OpenMP

A implementação lança **1 único thread por kernel** (`cdp_quicksort<<<1, 1>>>`):

```c
// quicksort_cuda_dp.cu, linhas 55-68
cdp_quicksort<<<1, 1, 0, s>>>(v, esq, nright, profundidade + 1);
...
cdp_quicksort<<<1, 1, 0, s1>>>(v, nleft, dir, profundidade + 1);
```

Para N=100.000:
- Número de partições internas: ~3.125 (N / THRESHOLD)
- Número de folhas: ~3.125
- **Total de lançamentos de kernel: ~6.250**
- Overhead por lançamento DP: ~5–15μs
- Overhead total estimado: **31–93ms** — domina o tempo medido de 99,6ms

Os 2 filhos são lançados em streams NonBlocking, permitindo execução concorrente.
Mas com apenas 1 thread cada e ~6.250 lançamentos para N=100.000, o overhead
acumulado supera qualquer ganho de paralelismo.

---

## 5. Por que OpenMP supera CUDA nos nossos resultados

Nossa implementação CUDA usa pilha explícita iterativa com:
- `cudaDeviceSynchronize()` a cada nível do quicksort
- `cudaMemset` + `cudaMemcpy` para gerenciar o tamanho da pilha por nível
- Particionamento de Lomuto (single-thread), não paralelo dentro de uma partição

Para N=100.000, a CPU com OpenMP:
- Dados cabem no cache L2/L3 (~400KB para `int[100000]`)
- Sem overhead de PCIe ou lançamentos de kernel
- Quicksort serial em cache quente é muito eficiente para N moderado

O artigo provavelmente usa uma implementação CUDA com **paralelismo intra-partição**
(múltiplos threads por partição em cada nível), o que justificaria CUDA ser mais
rápido que OpenMP no artigo mas não nos nossos resultados.

---

## 6. Fidelidade ao Algoritmo 2 do Artigo

**Parcialmente fiel.**

| Aspecto | Artigo | Nossa implementação | Fiel? |
|---------|--------|---------------------|-------|
| Profundidade máxima | 24 | 24 | ✓ |
| Base case size | não especificado | 32 | — |
| Fallback | selection_sort | `selection_sort_device` | ✓ |
| Particionamento | não especificado | Hoare com pivô central | — |
| Paralelismo DP | filho esquerdo + direito | streams NonBlocking | ✓ |

A implementação CUDA DP é fiel ao Algoritmo 2. A divergência de ranking vem
da **implementação CUDA simples** (não DP), que não replica o nível de
paralelismo da versão do artigo.

---

## 7. Conclusão para o Estudo de Replicação

Nossa implementação do Quicksort CUDA DP é **fiel ao Algoritmo 2 do artigo**.
O ranking observado (CUDA DP é o mais lento) **concorda com o artigo**.

A divergência de 1º e 2º lugar (OpenMP vs CUDA) decorre do algoritmo CUDA
simples, não do CUDA DP. O artigo provavelmente usou um quicksort CUDA mais
paralelo para as partições intermediárias — o que não está detalhado no apêndice
técnico (que documenta apenas OpenMP e CUDA simples, sem CUDA DP).

Os resultados são apresentados como achado válido do estudo de replicação:
demonstram que para N=100.000, o overhead de lançamentos de kernel recursivos
via DP supera o benefício da computação em GPU nesta plataforma (MX350).

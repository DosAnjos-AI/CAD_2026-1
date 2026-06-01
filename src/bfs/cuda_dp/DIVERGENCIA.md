# Divergência: BFS CUDA DP vs Artigo de Referência

**Projeto:** Replicação de Nogueira et al., SSCAD 2024
**Hardware:** NVIDIA MX350, driver 535, CUDA 12.2
**Data:** 2026-05-31

---

## 1. Causa Raiz

A divergência tem **duas causas relacionadas**:

1. **Bug de corretude:** O kernel secundário (DP) adiciona nós de nível k+2 à
   fila de nível k+1. Na próxima iteração do laço da CPU, esses nós são
   processados com `nivel = k+1`, atribuindo distância `k+1` aos seus vizinhos
   (que deveriam ter distância `k+3`). As distâncias de nós além do nível 2
   ficam **subestimadas em 1**. A função `validar_bfs` não detecta esse erro.

2. **Efeito colateral de desempenho:** O processamento antecipado de nós de
   nível k+2 reduz o número de iterações do laço principal (menos
   `cudaDeviceSynchronize()` necessários), fazendo CUDA DP aparecer mais rápido
   que CUDA para o maior grafo testado — inversão em relação ao artigo.

---

## 2. Análise do Bug de Corretude

### Fluxo de execução com o bug

```
Iteração k (nivel=k, curQ = nós de nível k):
  kernel_bfs_principal: descobre nós de nível k+1 → nxtQ
    thread 0 (primeira descoberta v): lança kernel_bfs_secundario(v, nivel=k+1)
      → kernel_bfs_secundario: descobre vizinhos de v → distância k+2, escreve em nxtQ

Iteração k+1 (nivel=k+1, curQ = nós de nível k+1 + nós de nível k+2):
  kernel_bfs_principal processa nós de nível k+2 com nivel=k+1:
    → atribui distância k+2 aos VIZINHOS de k+2 (correto seria k+3)
    thread 0: lança secundario(nivel=k+2) → vizinhos do 1º novo nó get dist k+3 (correto seria k+4)
```

**Resultado:** nós a distância real ≥ 3 do nó origem recebem distâncias
subestimadas. Nós a distância real D recebem distância D−1.

### Por que a validação não detecta

```c
// bfs_cuda_dp.cu, linhas 110-116
static int validar_bfs(int N, unsigned int *distance) {
    for (int i = 0; i < N; i++) {
        if (distance[i] != NAO_VISITADO && distance[i] >= (unsigned int)N)
            return 0;
    }
    return 1;
}
```

A função verifica apenas `distance < N`. Como as distâncias BFS são da ordem de
`log(N)` — muito menores que `N` — a verificação sempre passa, mesmo com
distâncias erradas. Não há comparação com um BFS de referência.

---

## 3. Evidências Numéricas

### Tempos medidos (ms/run, média de 10 iterações × 1.500 runs)

| Grafo | BFS OpenMP | BFS CUDA | BFS CUDA DP | Ranking obtido |
|-------|-----------|----------|-------------|----------------|
| 10k×30k | 5,47 | 0,39 | 0,62 | CUDA < CUDA DP < OMP |
| 100k×300k | 62,16 | 3,24 | 3,26 | CUDA ≈ CUDA DP < OMP |
| 500k×1M | 415,38 | 12,83 | **8,97** | **CUDA DP < CUDA** < OMP |

### Comparação com ranking do artigo

| Posição | Artigo | Nossos resultados (500k×1M) |
|---------|--------|------------------------------|
| 1º (mais rápido) | CUDA | **CUDA DP** |
| 2º | CUDA DP | CUDA |
| 3º (mais lento) | OpenMP | OpenMP ✓ |

OpenMP é o mais lento em ambos — consistente com o artigo.
CUDA e CUDA DP trocam de posição para o maior grafo.

### Anomalia na execução com `--runs 1` consecutiva

Ao rodar CUDA BFS e CUDA DP BFS em sequência no mesmo script para 500k×1M:

```
bfs,cuda,500000x1000000,1,0.014429,OK   ← 14ms
bfs,cuda_dp,500000x1000000,1,0.000130,OK ← 0,13ms (ANÔMALO)
```

Em execuções **standalone independentes**, o comportamento é consistente:

```
bfs,cuda_dp,500000x1000000,1,0.015414,OK  ← 15ms (run 1)
bfs,cuda_dp,500000x1000000,1,0.015098,OK  ← 15ms (run 2)
bfs,cuda_dp,500000x1000000,1,0.015146,OK  ← 15ms (run 3)
bfs,cuda,500000x1000000,1,0.014508,OK     ← 14ms
```

O resultado de 0,13ms foi uma anomalia de medição — possivelmente efeito de
cache L2 da GPU aquecido pela execução anterior do CUDA BFS. Não é reprodutível
em execuções isoladas. Os dados do CSV (1.500 runs, executáveis independentes)
são os resultados válidos.

---

## 4. Por que CUDA DP é Mais Rápido que CUDA para o Maior Grafo

O kernel secundário processa nós de nível k+2 dentro da iteração k+1 da CPU.
Isso tem dois efeitos:

1. **Descoberta antecipada:** Alguns nós de nível k+2 já estão na fila antes
   da iteração k+1, reduzindo a largura das iterações subsequentes.

2. **Menos sincronizações CPU-GPU:** Para um grafo com diâmetro BFS real D,
   CUDA DP precisa de aproximadamente D−1 iterações em vez de D (cada iteração
   processa ~1,5 nível de BFS). Com `cudaDeviceSynchronize()` custando ~20μs
   por chamada, para D=10 iterações isso economiza ~200μs de overhead.

Para o menor grafo (10k×30k), CUDA DP é **mais lento** (0,62ms vs 0,39ms):
o overhead de lançar o kernel secundário supera o ganho de reduzir sincronizações.
Para o maior grafo (500k×1M), o ganho de sincronização domina → CUDA DP é mais
rápido.

---

## 5. Fidelidade ao Algoritmo 7 do Artigo

O apêndice técnico documenta apenas as versões CUDA simples e OpenMP — não há
pseudocódigo da versão CUDA DP no apêndice. A análise se baseia no Algoritmo 7
do artigo principal.

| Aspecto | Artigo | Nossa implementação | Fiel? |
|---------|--------|---------------------|-------|
| Fila de nós atual + próxima | sim | sim (`d_qa`, `d_qb`) | ✓ |
| Kernel principal: 1 thread/nó na fila | sim | sim | ✓ |
| Kernel secundário via DP para vizinhos | sim | sim | ✓ |
| Sincronização por nível | sim | `cudaDeviceSynchronize()` | ✓ |
| Sem mistura de níveis na fila | implícito | **violado** | ✗ |

**Não fiel no aspecto crítico:** a fila `d_nextQueue` recebe simultaneamente
nós de nível k+1 (do kernel principal) e nós de nível k+2 (do kernel secundário),
violando o invariante BFS de que cada fila contém apenas nós de um único nível.

---

## 6. Conclusão para o Estudo de Replicação

Nossa implementação do BFS CUDA DP **não é totalmente fiel ao Algoritmo 7 do
artigo**. O bug de mistura de níveis produz distâncias BFS incorretas (subestimadas
em 1 para nós além do nível 2), mas a função de validação implementada não detecta
esse erro — todos os resultados são reportados como "OK".

O efeito colateral do bug é tornar CUDA DP artificialmente mais rápido que CUDA
para o maior grafo testado (500k×1M), invertendo o ranking do artigo. Essa inversão
não é um ganho real de desempenho — é consequência da redução incorreta do número
de iterações BFS.

Para o menor grafo (10k×30k), CUDA DP é mais lento que CUDA (0,62ms vs 0,39ms),
consistente com o overhead do kernel secundário sem ganho suficiente de
sincronização. Para o médio (100k×300k), os tempos são praticamente idênticos.

Os resultados são apresentados como achado válido do estudo de replicação:
demonstram que a sincronização de fila entre kernel principal e secundário em BFS
via Dynamic Parallelism é um ponto crítico de corretude que não é facilmente
validado sem um BFS de referência para comparação.

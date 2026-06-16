# DECISOES_TECNICAS.md — Referência de Implementação CAD_2026_v3

## Leitura obrigatória antes de gerar qualquer prompt

Este documento concentra todas as decisões técnicas de implementação definidas em conjunto.
Deve ser lido integralmente antes de gerar cada prompt para o Claude Code.
Complementa o CLAUDE.md — não repete regras gerais, foca em decisões de implementação.

---

## Decisões gerais de implementação

### Tipo de dado
- `int32_t` em todos os algoritmos e APIs
- Incluir `<stdint.h>` em arquivos `.c`

### Medição de tempo
- `clock_gettime(CLOCK_MONOTONIC)` em C puro e OpenMP
- `cudaEvent_t` com `cudaEventElapsedTime` em CUDA e CUDA DP
- Unidade: segundos com precisão de nanosegundos
- `tempo_s` = média aritmética das execuções internas por iteração

### Geração de dados aleatórios
- `srand(42)` — semente fixa para reprodutibilidade entre APIs
- `rand()` para geração de valores inteiros
- Vetor regenerado a cada iteração com a mesma semente

### Saída CSV
- Formato: `algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos`
- Decimais com vírgula: substituição manual de '.' por ',' na string de saída
- Saída para `stdout` — append feito pelo `run_all.sh`
- Locale: não depender de `setlocale` — fazer substituição direta do ponto por vírgula

### Protocolo de execução
- 3 warmup descartados (sem saída, sem CSV)
- 10 iterações medidas
- Cada iteração: 10 execuções internas, tempo_s = média delas
- Total: 130 execuções por combinação

### Validação de corretude
- Comparação exata elemento a elemento (int32, sem epsilon)
- Oráculo para vetores: `qsort` da libc em cópia do vetor original
- Oráculo para BFS: BFS sequencial sobre o mesmo grafo
- Oráculo para Floyd-Warshall: versão sequencial sobre a mesma matriz
- `corretude=1` se idêntico, `corretude=0` caso contrário

### Argumentos do binário
- Recebe `tamanho` como único argumento obrigatório
- Sem argumentos adicionais — tudo detectado internamente ou fixo no código
- Validar que N é potência de 2 para algoritmos que exigem (Bitonic Sort)

---

## Bitonic Sort

### Algoritmo sequencial (CPU)
- Implementação iterativa com dois loops aninhados: `k` (tamanho do bloco) e `j` (passo)
- `k` vai de 2 até N duplicando; `j` vai de `k/2` até 1 dividindo
- Direção de ordenação determinada por `(i & k) == 0`
- `compare_and_swap` inline — sem chamada de função separada
- Sem recursão, sem padding
- Estrutura do arquivo: includes → compare_and_swap → bitonic_sort → main

### Algoritmo OpenMP
- Paralelizar o loop externo de pares com `#pragma omp parallel for`
- Atenção: sincronização entre fases — barrier implícito no fim de cada `parallel for`
- Número de threads: `NUM_THREADS` definido via `-DNUM_THREADS=$(nproc)` na compilação
- Sem `omp_set_num_threads` hardcoded — usar a macro

### Algoritmo CUDA
- Kernel iterativo: um thread por par de comparação (N/2 threads total)
- Truque XOR: `ixj = i ^ j` determina o parceiro de comparação de cada thread
- Direção: `(i & k) == 0` determina se deve ordenar crescente ou decrescente
- Lançamento: `dim3 blocks(N/BLOCK_SIZE/2)`, `dim3 threads(BLOCK_SIZE)`
- `BLOCK_SIZE = 256` — melhor resultado empírico na literatura (128-256)
- Sincronização entre passos via múltiplos lançamentos de kernel (não __syncthreads global)
- Sem shared memory — implementação global memory first (KISS)
- `threads_blocos` no CSV: formato `256xNBLOCKS`

### Algoritmo CUDA DP
- Kernel pai lança kernels filhos para cada fase do bitonic sort
- Cada fase (par k,j) é um kernel filho lançado com `cudaLaunchKernel` ou chamada direta
- Compilar com `-rdc=true` obrigatório
- Justificativa acadêmica: estrutura recursiva hierárquica — cada nível de merge é independente
- Nota honesta: CUDA DP não garante speedup sobre CUDA flat para Bitonic Sort
  (estrutura regular não se beneficia tanto de DP quanto estruturas irregulares)
- `threads_blocos` no CSV: formato `256xNBLOCKS_dp`

---

## Merge Sort

### Algoritmo sequencial (CPU)
- Implementação recursiva clássica top-down com buffer auxiliar
- `merge` iterativo (não recursivo) para evitar stack overflow em N grandes
- Buffer auxiliar alocado uma vez fora da recursão — passado como parâmetro
- Estrutura: includes → merge → merge_sort → main

### Algoritmo OpenMP
- Paralelizar as chamadas recursivas com `#pragma omp task`
- Cutoff para serializar subproblemas pequenos (N < CUTOFF) — CUTOFF = 1024
- `#pragma omp taskwait` antes do merge
- Threads via `NUM_THREADS`

### Algoritmo CUDA
- Bottom-up iterativo: começa com subarrays de tamanho 1, dobra até N
- Cada passo de merge é um kernel independente
- Buffer auxiliar na device memory (ping-pong entre dois buffers)
- `BLOCK_SIZE = 256`
- `threads_blocos` no CSV: `256xNBLOCKS`

### Algoritmo CUDA DP
- Kernel pai divide o array e lança kernels filhos para ordenar cada metade
- Merge final feito no kernel pai após sincronização dos filhos
- Mapeamento natural de divide-and-conquer para CUDA DP
- `-rdc=true` obrigatório
- `threads_blocos` no CSV: `256xNBLOCKS_dp`

---

## BFS

### Grafo
- Tipo: Erdős–Rényi aleatório com grau médio 16
- Representação: CSR (Compressed Sparse Row) — arrays `row_ptr` e `col_idx` de `int32_t`
- Geração: para cada par (u,v), aresta existe com probabilidade p = 16.0/V
- Semente: `srand(42)` — mesma semente em todas as APIs
- Vértice fonte: sempre vértice 0
- Grafo não-dirigido: cada aresta adicionada nos dois sentidos

### Algoritmo sequencial (CPU)
- BFS por nível com fila simples (array circular ou dois arrays de fronteira)
- Array `dist[V]` inicializado com -1, dist[0] = 0
- Dois arrays de fronteira: `frontier_atual` e `frontier_prox`
- Sem queue dinâmica — arrays pré-alocados de tamanho V

### Algoritmo OpenMP
- Paralelizar o processamento da fronteira atual com `#pragma omp parallel for`
- `dist` atualizado com `#pragma omp critical` ou `__sync_bool_compare_and_swap`
- Usar comparação atômica para evitar race condition na atualização de distâncias
- Sincronização entre níveis via barrier implícito

### Algoritmo CUDA
- Kernel por nível: um thread por vértice na fronteira
- Arrays na device: `dist`, `frontier`, `visited` (int32_t)
- Loop no host: enquanto fronteira não vazia, lançar kernel
- Fronteira gerenciada com flag booleano de `frontier_ativa`
- `BLOCK_SIZE = 256`
- `threads_blocos` no CSV: `256xNBLOCKS`

### Algoritmo CUDA DP
- Kernel pai processa a fronteira atual
- Para cada vértice ativo na fronteira, lança kernel filho para expandir vizinhos
- Justificativa natural: irregularidade da fronteira BFS — DP foca recursos nos vértices ativos
- `-rdc=true` obrigatório
- `threads_blocos` no CSV: `256xNBLOCKS_dp`

### Corretude BFS
- Oráculo: BFS sequencial sobre o mesmo grafo (mesmo CSR, mesma semente)
- Comparar array `dist[]` elemento a elemento

---

## Floyd-Warshall

### Grafo
- Matriz de adjacência densa `int32_t dist[V][V]`
- Pesos aleatórios no intervalo [1, 1000] com `srand(42)`
- Diagonal principal = 0
- INF = 1e9 (sentinela para ausência de aresta)
- Densidade: ~50% das arestas presentes (para cada par i≠j, aresta com prob 0.5)
- Matriz alocada como array 1D de tamanho V*V (row-major) — melhor coalescência

### Algoritmo sequencial (CPU)
- Triple loop clássico: k (intermediário) → i (origem) → j (destino)
- `if (dist[i*V+k] + dist[k*V+j] < dist[i*V+j])` atualiza
- Verificar overflow: usar `(dist[i*V+k] < INF && dist[k*V+j] < INF)` antes de somar
- Sem otimizações — sequencial puro

### Algoritmo OpenMP
- Paralelizar loops i e j com `#pragma omp parallel for collapse(2)`
- Loop k deve permanecer sequencial (dependência de dados entre iterações de k)
- Threads via `NUM_THREADS`

### Algoritmo CUDA
- Kernel por iteração de k: V*V threads, um por par (i,j)
- Lançamento: `dim3 blocks(V/BLOCK_SIZE, V/BLOCK_SIZE)`, `dim3 threads(BLOCK_SIZE, BLOCK_SIZE)`
- `BLOCK_SIZE = 16` — mais adequado para kernel 2D (16*16 = 256 threads/bloco)
- Loop k no host, kernel lançado V vezes
- `threads_blocos` no CSV: `16x(V/16)²`

### Algoritmo CUDA DP (R-Kleene / divide-and-conquer)
- Dividir a matriz em quadrantes recursivamente
- Kernel pai processa quadrantes independentes em paralelo via kernels filhos
- Base da recursão: submatriz pequena (V <= 32) — resolver diretamente no kernel
- Justificativa: quadrantes independentes em cada nível = paralelismo hierárquico natural
- `-rdc=true` obrigatório
- Nota honesta: pode não superar CUDA flat para FW — documentar como resultado válido
- `threads_blocos` no CSV: `16x(V/16)²_dp`

### Corretude Floyd-Warshall
- Oráculo: versão sequencial sobre a mesma matriz (mesma semente)
- Comparar matriz resultado elemento a elemento (int32, exato)

---

## Referências de implementação consultadas

- Bitonic Sort CUDA iterativo: gist.github.com/mre/1392067
- Bitonic Sort CPU iterativo: geeksforgeeks.org/dsa/bitonic-sort
- Merge Sort CUDA bottom-up: moderngpu.github.io/mergesort.html
- BFS CUDA por nível: Harish & Narayanan (CUDA BFS, 2007)
- BFS CUDA DP: dl.acm.org/doi/10.1145/2833179.2833189
- Floyd-Warshall CUDA: github.com/OlegKonings/CUDA_Floyd_Warshall_
- Floyd-Warshall R-Kleene: arxiv.org/abs/2310.03983
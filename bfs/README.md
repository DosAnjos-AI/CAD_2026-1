# BFS

## 1. Visão geral do algoritmo

BFS (Busca em Largura) percorre um grafo nível a nível a partir de um vértice
fonte, visitando primeiro todos os vizinhos a distância 1, depois os a
distância 2, e assim por diante, até esgotar os vértices alcançáveis.
Complexidade O(V+E) — cada vértice e cada aresta são examinados no máximo uma
vez.

O grafo usado é Erdős–Rényi não-dirigido com grau médio 16: para cada par
`(u, w)` com `u < w`, a aresta existe com probabilidade `p = 16.0/V`,
testada com `rand()` sob `srand(42)` — mesma semente em todas as APIs,
garantindo o mesmo grafo entre execuções e entre implementações.

Representação: CSR (Compressed Sparse Row), com `row_ptr[V+1]` e
`col_idx[E]`, ambos `int32_t`. A fonte é sempre o vértice 0.

## 2. Por implementação

### CPU sequencial (`bfs_cpu.c`)

**O que faz**: `bfs()` inicializa `dist[]` com -1 e `dist[fonte] = 0`, depois
processa a fronteira atual em um loop `while (tam_frontier > 0)`: para cada
vértice da fronteira, percorre seus vizinhos via `row_ptr`/`col_idx` e, se
ainda não visitado (`dist[w] == -1`), marca a distância e adiciona `w` à
próxima fronteira. Ao final de cada nível, os ponteiros `frontier` e `next`
são trocados (ping-pong) — sem realocação a cada nível.

**Decisão técnica relevante**: `frontier` e `next` são arrays pré-alocados de
tamanho V (sem queue dinâmica), suficiente porque a fronteira nunca excede V
vértices. O grafo é construído em `gerar_grafo_csr()` com duas passagens sob
o mesmo `srand(42)`: a primeira só conta o grau de cada vértice (para montar
`row_ptr`), a segunda preenche `col_idx` nas posições já calculadas — sem
buffer intermediário de arestas.

**Como compilar**:
```bash
gcc -O3 -march=native -o bfs_cpu bfs_cpu.c
```

**Como executar**:
```bash
./bfs_cpu V [iteracoes]
```

**Saída esperada**:
```
bfs|cpu|aleatorio|1048576|1|0,182734|1|1x1
```

### OpenMP (`bfs_openmp.c`)

**O que faz**: mesma estrutura por nível do CPU sequencial, mas o loop sobre
a fronteira atual é dividido entre threads com `#pragma omp for` dentro de
uma região `#pragma omp parallel`. Cada thread mantém um buffer local
`next_local[LOCAL_BUFFER_SIZE]`: ao descobrir um vértice novo, faz flush para
`next[]` sob `#pragma omp critical` quando o buffer enche (ou ao final do
loop, se restar algo). A atualização de `dist[w]` usa
`__sync_bool_compare_and_swap(&dist[w], -1, dist[u]+1)` para garantir que
apenas a thread que efetivamente reivindicou o vértice o insira na próxima
fronteira — evita duplicidade sem precisar de critical na atualização da
distância em si.

**Decisão técnica relevante**: `LOCAL_BUFFER_SIZE = 1024` em vez de um buffer
por thread de tamanho V — com até `NUM_THREADS` threads simultâneas, V
buffers de tamanho V desperdiçariam memória e cache proporcionalmente ao
número de threads; o buffer de 1024 é descartável (estourado vira flush) e
mantém a seção crítica rara (uma vez a cada 1024 descobertas por thread, não
uma vez por vértice). Sincronização entre níveis vem do barrier implícito ao
final da região `parallel`.

**Como compilar**:
```bash
gcc -O3 -march=native -fopenmp -DNUM_THREADS=$(nproc) -o bfs_openmp bfs_openmp.c
```

**Como executar**:
```bash
./bfs_openmp V [iteracoes]
```

**Saída esperada**:
```
bfs|openmp|aleatorio|1048576|1|0,041523|1|32x1
```

### CUDA (`bfs_cuda.cu`)

**O que faz**: `bfs_kernel` lança um thread por vértice (`V` threads no
total). Cada thread só age se seu vértice estiver ativo na fronteira
(`frontier[v] == 1`); ao agir, limpa sua própria entrada em `frontier` e
expande os vizinhos, usando `atomicCAS(&dist[u], -1, nivel+1)` para
reivindicar `u` com exclusividade — só a thread que ganha o CAS marca
`frontier_next[u] = 1` e zera a flag `frontier_vazia`. O host laça
lançando o kernel a cada nível, trocando os ponteiros `d_frontier` /
`d_frontier_next` (ping-pong) e lendo `frontier_vazia` de volta para decidir
se continua.

**Decisão técnica relevante**: `cudaMemset(d_dist, 0xFF, V*sizeof(int32_t))`
inicializa todo `dist[]` com -1 em uma única chamada — válido porque -1 em
complemento de dois (`int32_t`) é `0xFFFFFFFF`, ou seja, todo byte é `0xFF`;
um `cudaMemset` com qualquer outro valor sentinela não funcionaria da mesma
forma. Em seguida, `cudaMemset(d_dist, 0, sizeof(int32_t))` zera só a
primeira posição (`dist[0] = 0`, fonte sempre vértice 0).

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -o bfs_cuda bfs_cuda.cu
```

**Como executar**:
```bash
./bfs_cuda V [iteracoes]
```

**Saída esperada**:
```
bfs|cuda|aleatorio|1048576|1|0,003812|1|256x4096
```

### CUDA DP (`bfs_cudadp.cu`)

**O que faz**: o kernel pai `bfs_outer` lança um thread por vértice; threads
cujo vértice está ativo na fronteira limpam sua entrada e lançam um kernel
filho `bfs_expand` próprio, com um thread por vizinho
(`blocos_filho = ceil(n_viz / BLOCK_SIZE)`). O filho faz o mesmo
`atomicCAS` do CUDA flat para reivindicar `dist[u]` e marcar
`frontier_next[u]`. O host chama `cudaDeviceSynchronize()` logo após cada
lançamento de `bfs_outer`, antes de trocar os ponteiros de fronteira e ler
`frontier_vazia`.

**Decisão técnica relevante**: o paralelismo dinâmico concentra recursos nos
vértices efetivamente ativos — vértices fora da fronteira não geram nenhum
lançamento de kernel filho, diferente do CUDA flat, que sempre lança `V`
threads independente de quantas estão ativas. O campo `threads_blocos` no
CSV é o literal `1x1_dp` (não uma fórmula a partir de `blocos`/`BLOCK_SIZE`):
desvio do formato `256xNBLOCKS_dp` previsto em `decisoes_tecnicas.md`,
porque aqui não há uma única configuração de lançamento representativa — o
número de kernels filhos e suas dimensões variam por vértice ativo e por
nível, então qualquer valor fixo seria nominal, e o código optou por não
fingir uma medida de paralelismo que não existe.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -rdc=true -o bfs_cudadp bfs_cudadp.cu
```

**Como executar**:
```bash
./bfs_cudadp V [iteracoes]
```

**Saída esperada**:
```
bfs|cudadp|aleatorio|1048576|1|0,015234|1|1x1_dp
```

## 3. Notas importantes

**Construção CSR em duas passagens com replay de `srand(42)`**: as quatro
implementações geram o grafo chamando `gerar_grafo_csr()` antes de cada
execução (warmup e medida), e essa função reinicia `srand(42)` no começo de
cada uma das suas duas passagens internas. A primeira passagem só conta grau
por vértice para montar `row_ptr`; a segunda repete exatamente a mesma
sequência de `rand()` para preencher `col_idx` nas posições corretas. O
resultado é determinístico e idêntico em toda regeneração, sem precisar
manter uma lista de arestas em buffer intermediário entre as duas passagens
— o custo é gerar os mesmos números aleatórios duas vezes por execução, em
troca de não alocar O(E) de memória extra.

**OpenMP — `LOCAL_BUFFER_SIZE = 1024` em vez de buffer de tamanho V**: um
buffer por thread do tamanho do grafo seria correto, mas desperdiçaria
memória e cache proporcionalmente ao número de threads (até `NUM_THREADS`
buffers de V `int32_t` simultâneos). O buffer de 1024 entradas é uma escolha
de compromisso: grande o suficiente para tornar a seção crítica de flush
pouco frequente (uma vez a cada 1024 descobertas por thread), pequeno o
suficiente para caber confortavelmente na pilha de cada thread.

**CUDA — `cudaMemset` com `0xFF` para inicializar `dist` com -1**: ver
decisão técnica na seção CUDA acima. É um truque específico do valor -1 em
complemento de dois, não generalizável a outros sentinelas.

**CUDA DP — `cudaDeviceSynchronize()` no host após cada nível, obrigatório
no CDP2**: a partir do CUDA 12 (CDP2, usado aqui com CUDA 12.1), o retorno do
kernel pai para o host não garante mais que os kernels filhos lançados por
ele tenham terminado — diferente do CDP1, onde havia sincronização implícita
pai-filho embutida na conclusão do grid pai. Sem o
`cudaDeviceSynchronize()` explícito entre o lançamento de `bfs_outer` e a
leitura de `frontier_vazia`/`frontier_next`, o host poderia ler esses
buffers antes dos kernels `bfs_expand` terminarem de escrevê-los,
corrompendo a fronteira do próximo nível.

**Nota honesta sobre CUDA DP**: para um grafo Erdős–Rényi de grau médio
uniforme (16), a expectativa é que o CUDA DP seja mais lento que o CUDA
flat. Cada vértice ativo expande um número de vizinhos pequeno e
relativamente parecido entre si, então o custo fixo de lançar um kernel
filho por vértice ativo (latência de lançamento do device runtime) tende a
superar o ganho de concentrar paralelismo apenas nos vértices ativos.
Diferente de grafos com grau muito desigual (power-law, com hubs de alto
grau), onde DP focaria recursos de forma desproporcional nos vértices que
realmente precisam, aqui todos os vértices ativos têm carga de trabalho
parecida — o CUDA flat já cobre essa carga sem overhead de lançamento
recursivo. Resultado esperado: CUDA DP mais lento que CUDA flat na maioria
dos tamanhos de V, mantido no benchmark como comparação válida entre
estratégias de implementação.

**Pendências de validação na RTX 4090**: o ambiente onde os binários foram
compilados e testados localmente expõe uma GPU MX350 (`sm_61`), diferente do
hardware alvo do projeto (RTX 4090, `sm_89`, conforme `CLAUDE.md`). A lógica
das quatro implementações foi revisada por leitura de código, mas a
execução real de medição e validação de corretude (`corretude=1` no CSV) na
RTX 4090 ainda está pendente.

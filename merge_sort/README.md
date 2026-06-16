# Merge Sort

## 1. Visão geral do algoritmo

Merge Sort é um algoritmo de ordenação por comparação, baseado em divisão e
conquista: o vetor é dividido recursivamente em metades até subvetores
unitários, e a ordenação final emerge da intercalação (merge) sucessiva
dessas metades já ordenadas. Complexidade O(n log n) no caso médio, melhor e
pior caso — não depende da distribuição dos dados de entrada.

Diferente do Bitonic Sort, não exige que N seja potência de 2: a divisão
`mid = left + (right - left) / 2` funciona para qualquer tamanho, e o merge
trata subvetores de tamanhos desiguais sem ajuste especial.

É adequado para paralelização porque as duas metades de cada nível da
recursão são completamente independentes entre si — não há dependência de
dados até o momento do merge, que é o único ponto de sincronização. Isso
mapeia diretamente para tasks (OpenMP), kernels independentes por passo
(CUDA bottom-up) ou kernels filhos (CUDA DP).

## 2. Por implementação

### CPU sequencial (`merge_cpu.c`)

**O que faz**: implementação top-down clássica. `merge_sort` recursivo divide
o intervalo `[left, right]` em duas metades, ordena cada uma recursivamente e
intercala o resultado com `merge`. A função `merge` é iterativa (três loops
`while`, sem recursão) e usa um buffer auxiliar `tmp` alocado uma única vez
no `main` e passado por parâmetro em toda a recursão — evita malloc repetido
a cada chamada.

**Decisão técnica relevante**: o buffer auxiliar único (`tmp`, `arr`, `ref`)
é alocado fora da recursão e reutilizado em todas as iterações e execuções
internas, custo de alocação não entra na medição de tempo. O oráculo de
corretude é gerado com `qsort` da libc sobre uma cópia (`ref`) do vetor
original, comparado ao resultado de `arr` após a ordenação.

**Como compilar**:
```bash
gcc -O3 -march=native -o merge_cpu merge_cpu.c
```

**Como executar**:
```bash
./merge_cpu N [iteracoes]
```

**Saída esperada**:
```
merge_sort|cpu|aleatorio|1048576|1|0,734521|1|1x1
```

### OpenMP (`merge_openmp.c`)

**O que faz**: mesma estrutura recursiva do CPU sequencial, mas a divisão em
`merge_sort_omp` lança cada metade como `#pragma omp task`, com
`#pragma omp taskwait` antes do merge para garantir que ambas as metades
estejam ordenadas. Abaixo de `CUTOFF = 1024` elementos, a recursão cai para
`merge_sort_seq` (mesma lógica do CPU, sem overhead de tasks). A chamada
inicial é feita dentro de uma região `#pragma omp parallel` com
`#pragma omp single`, padrão necessário para que as tasks sejam criadas e
distribuídas pelo time de threads.

**Decisão técnica relevante**: `omp_set_num_threads(NUM_THREADS)` é chamado
uma única vez no início do `main` (não a cada iteração), com `NUM_THREADS`
vindo de `-DNUM_THREADS=$(nproc)` na compilação. O cutoff de 1024 evita
overhead de criação de tasks para subproblemas pequenos, onde o custo de
gerenciamento supera o ganho de paralelismo. O campo `threads_blocos` no CSV
usa `omp_get_max_threads()` (não a macro diretamente), refletindo o valor
efetivo do runtime.

**Como compilar**:
```bash
gcc -O3 -march=native -fopenmp -DNUM_THREADS=$(nproc) -o merge_openmp merge_openmp.c
```

**Como executar**:
```bash
./merge_openmp N [iteracoes]
```

**Saída esperada**:
```
merge_sort|openmp|aleatorio|1048576|1|0,098234|1|32x1
```

### CUDA (`merge_cuda.cu`)

**O que faz**: bottom-up iterativo. O host laça `sub_size` de 1 até N,
dobrando a cada passo, e lança um kernel por passo (`merge_kernel`). Os
buffers `d_arr`/`d_tmp` funcionam em ping-pong: cada kernel lê do buffer
atual e escreve no outro, trocando os papéis a cada passo. Ao final, se o
número de passos for ímpar, o resultado está em `d_tmp` e é copiado de volta
para `d_arr` via `cudaMemcpyDeviceToDevice`.

**Decisão técnica relevante**: dentro de cada passo de merge, em vez de um
thread fazer o merge serial de um par completo de subarrays (abordagem mais
simples), cada thread calcula independentemente, via `co_rank` (merge path,
Kirk & Hwu), qual elemento da entrada ocupa sua posição de saída. Isso
permite N/BLOCK_SIZE blocos de BLOCK_SIZE threads cobrindo o vetor inteiro
por passo, com paralelismo proporcional ao tamanho do vetor e não ao número
de pares de subarrays — relevante porque nos últimos passos do bottom-up há
poucos pares (no limite, um único merge final de N/2 com N/2), e um thread
por merge desperdiçaria a maior parte do hardware. Custo: busca binária por
thread (`co_rank`) em vez de acesso sequencial direto.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -o merge_cuda merge_cuda.cu
```

**Como executar**:
```bash
./merge_cuda N [iteracoes]
```

**Saída esperada**:
```
merge_sort|cuda|aleatorio|1048576|1|0,004123|1|256x4096
```

### CUDA DP (`merge_cudadp.cu`)

**O que faz**: `merge_sort_dp` é um kernel que se auto-relança: divide o
intervalo em duas metades e lança dois kernels filhos (`merge_sort_dp<<<1,1>>>`
cada), um por metade, seguidos de um terceiro kernel (`merge_tail_kernel`)
lançado em `cudaStreamTailLaunch`, responsável pelo merge final. Abaixo de
`CUTOFF = 1024`, a recursão para e `merge_sort_seq_device` ordena o
subintervalo localmente, em uma única thread.

**Decisão técnica relevante**: todos os kernels da árvore de recursão
(pai, filhos e tail) são lançados como `<<<1,1>>>` — uma thread por nó da
recursão, já que o paralelismo aqui vem da árvore de chamadas (divide and
conquer), não de threads paralelas dentro de um nó. O campo `threads_blocos`
no CSV (`256x<blocos>_dp`) é calculado a partir de N e BLOCK_SIZE mas **não
corresponde** à configuração real de lançamento dos kernels da recursão, que
é sempre `<<<1,1>>>` — é um valor nominal mantido por consistência de
formato com as demais linhas do CSV, não uma medida de paralelismo efetivo
desta implementação.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -rdc=true -o merge_cudadp merge_cudadp.cu
```

**Como executar**:
```bash
./merge_cudadp N [iteracoes]
```

**Saída esperada**:
```
merge_sort|cudadp|aleatorio|1048576|1|0,021456|1|256x4096_dp
```

## 3. Notas importantes

**`cudaStreamTailLaunch` no lugar de `cudaDeviceSynchronize()`**: a versão
planejada em `decisoes_tecnicas.md` previa o kernel pai sincronizar com os
filhos e então fazer o merge ele mesmo. A implementação real usa
`cudaStreamTailLaunch` (CDP2, disponível a partir do CUDA 11.6, usado aqui em
CUDA 12.1): o kernel de merge é agendado nesse stream especial, que só
inicia execução quando o grid atual e toda a sua árvore de descendentes
tiverem terminado. Isso elimina a necessidade de sincronização explícita
dentro do device, que em CDP2 é mais restrita e custosa que em CDP1 (CUDA
< 12 chegava a proibir certas formas de sync filho-pai em versões recentes
do runtime). A escolha simplifica o kernel pai para apenas três lançamentos
assíncronos, sem bloqueio.

**`merge_sort_seq_device` iterativo, não recursivo**: a versão de base da
recursão (abaixo do cutoff) foi implementada como bottom-up iterativo
(`for (width = 1; width < n; width <<= 1)`), e não como a recursão top-down
usada no CPU e no OpenMP. O motivo é prático: o linker do CUDA (`nvlink`) não
consegue determinar estaticamente o tamanho de pilha necessário para uma
função `__device__` recursiva, e emite warning de stack size potencialmente
insuficiente em tempo de execução para profundidades de recursão maiores.
A versão iterativa tem uso de pilha constante e elimina o warning.

**Co-rank (merge path) no CUDA flat**: ver decisão técnica na seção CUDA
acima. A escolha sobre merge serial (um thread por merge completo) se deve à
distribuição desigual de trabalho do bottom-up: nos primeiros passos há
muitos pares pequenos (bom paralelismo natural), mas nos últimos passos há
poucos pares grandes — no último passo, um único merge de tamanho N/2 + N/2.
Sem co-rank, esse último passo seria executado por uma única thread,
anulando o paralelismo justamente na fase mais cara. Com co-rank, todo
elemento de saída é computado por uma thread distinta em qualquer passo,
mantendo ocupação cheia da GPU do início ao fim do algoritmo.

**Pendências de validação**: as implementações CUDA e CUDA DP ainda não
foram executadas e validadas na RTX 4090 (hardware alvo do projeto). A
lógica foi revisada por leitura de código, mas corretude (`corretude=1` no
CSV) e ausência de erros de runtime (limite de lançamentos pendentes do
device runtime, uso de memória, etc.) seguem pendentes de execução real.

**Nota honesta sobre CUDA DP**: a expectativa é que o overhead de lançamento
de kernels recursivos (cada nó da árvore de divisão gera 3 lançamentos de
kernel: 2 filhos + 1 tail) supere o ganho de paralelismo hierárquico em
relação ao CUDA flat bottom-up, especialmente para N grande, onde a árvore
de recursão chega a milhares de níveis de profundidade até o cutoff de 1024.
Diferente do BFS (onde a irregularidade da fronteira justifica DP) ou do
Floyd-Warshall R-Kleene (quadrantes independentes), o Merge Sort tem
divisão perfeitamente balanceada e regular — o CUDA flat já explora esse
paralelismo sem o custo de lançamento recursivo. Resultado esperado: CUDA DP
mais lento que CUDA flat para a maioria dos tamanhos de N, mantido no
benchmark como comparação válida entre estratégias de implementação.

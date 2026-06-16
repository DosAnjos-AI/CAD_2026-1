# Bitonic Sort

## Visão geral do algoritmo

Bitonic Sort é um algoritmo de ordenação por comparação baseado na construção e desfazimento
sucessivo de sequências bitônicas (sequências que crescem e depois decrescem, ou vice-versa).
Complexidade O(n log²n) — pior que os O(n log n) de algoritmos baseados em comparação geral,
mas com uma característica que compensa em hardware paralelo: o padrão de comparações é fixo e
conhecido antecipadamente, independente dos dados de entrada.

Exige que N seja potência de 2 — a estrutura recursiva de divisão em blocos `k` e passos `j`
depende disso para que cada bloco se divida exatamente pela metade em cada nível.

Adequação a GPU: como o padrão de comparações é estático (não depende de valores), todas as
comparações dentro de um mesmo passo `(k, j)` são independentes entre si e podem ser feitas em
paralelo sem nenhum branch condicionado a dados — apenas a posição `i` decide o que comparar,
nunca o valor. Isso elimina divergência de warp por dado e permite mapear cada comparação
diretamente a uma thread.

## CPU sequencial — `bitonic_cpu.c`

**O que faz**: implementação iterativa com três loops aninhados — `k` (tamanho do bloco
bitônico, dobra de 2 até N), `j` (distância de comparação, divide de `k/2` até 1) e `i` (índice
percorrendo todo o vetor). Para cada `i`, calcula `ij = i ^ j` e só compara/troca se `ij > i`
(evita comparar o mesmo par duas vezes). A direção (crescente/decrescente) é decidida por
`(i & k) == 0`.

**Decisão técnica relevante**: a comparação e troca (`compare_and_swap`) está inline dentro do
loop `i`, sem função separada — evita overhead de chamada em um trecho executado O(n log²n)
vezes. Sem recursão e sem padding do vetor (N já é validado como potência de 2 antes de
ordenar).

**Como compilar**:
```bash
gcc -O3 -march=native -o bitonic_cpu bitonic_cpu.c
```

**Como executar**:
```bash
./bitonic_cpu N [iteracoes]
```

**Saída esperada**:
```
bitonic_sort|cpu|aleatorio|1048576|1|0,823451|1|1x1
```

## OpenMP — `bitonic_openmp.c`

**O que faz**: mesma lógica de três loops do CPU sequencial. Apenas o loop interno de índices
`i` é paralelizado com `#pragma omp parallel for`; os loops `k` e `j` permanecem sequenciais no
thread principal.

**Decisão técnica relevante**: a sincronização entre passos `(k, j)` depende inteiramente do
barrier implícito ao final de cada `parallel for` — não há barrier explícito no código, pois o
próprio fim da região paralela já garante que todas as threads concluam o passo atual antes do
laço externo avançar para o próximo `j`. Número de threads fixado por `omp_set_num_threads
(NUM_THREADS)`, onde `NUM_THREADS` é definido via macro de compilação (não hardcoded no
arquivo).

**Como compilar**:
```bash
gcc -O3 -march=native -fopenmp -DNUM_THREADS=$(nproc) -o bitonic_openmp bitonic_openmp.c
```

**Como executar**:
```bash
./bitonic_openmp N [iteracoes]
```

**Saída esperada** (`threads_blocos` reflete `omp_get_max_threads()`, não um valor fixo):
```
bitonic_sort|openmp|aleatorio|1048576|1|0,312045|1|32x1
```

## CUDA — `bitonic_cuda.cu`

**O que faz**: kernel `bitonic_step` lançado uma vez por passo `(k, j)` — o loop `k, j` roda no
host, cada iteração dispara um kernel com `N/2` threads. Cada thread cuida de exatamente um par
de comparação.

**Decisão técnica relevante**: ao invés do mapeamento `tid → i` direto com checagem `ij > i`
(que desperdiçaria metade das threads, já usado no CPU/OpenMP), o índice `i` é calculado a
partir do `tid` global de forma que o bit na posição `j` de `i` seja sempre zero:
`i = (tid / j) * (j * 2) + (tid % j)`. Isso garante que `parceiro = i ^ j` seja sempre maior que
`i`, eliminando a necessidade da checagem condicional e mantendo as `N/2` threads lançadas todas
ativas e úteis. Sincronização entre passos é feita por lançamentos sucessivos de kernel na
stream default — não há `__syncthreads()` global nem cooperative groups; a serialização entre
kernels consecutivos na mesma stream garante a ordem correta. Sem shared memory — implementação
direta em memória global (KISS), sem otimização de localidade entre passos. `BLOCK_SIZE = 256`.
Lançamento: `blocos = N / 2 / BLOCK_SIZE`, `dim3 threads(BLOCK_SIZE)`.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_XX -o bitonic_cuda bitonic_cuda.cu
```

**Como executar**:
```bash
./bitonic_cuda N [iteracoes]
```

**Saída esperada**:
```
bitonic_sort|cuda|aleatorio|1048576|1|0,002981|1|256x2048
```

## CUDA DP — `bitonic_cudadp.cu`

**O que faz**: kernel pai `bitonic_sort_dp<<<1,1>>>` executado por uma única thread, que
controla inteiramente no device o loop `(k, j)` e lança o kernel filho `bitonic_step`
(idêntico ao da versão CUDA flat) uma vez por passo, sem retorno ao host entre passos.

**Decisão técnica relevante**: compilado com `-rdc=true` (Relocatable Device Code), obrigatório
para lançamento de kernels a partir do device. Sob CDP2 (modelo padrão a partir do CUDA 12.x,
em uso neste projeto com CUDA 12.1), lançamentos sucessivos feitos pela mesma thread sem stream
explícito caem na stream default por-thread, que serializa a execução na ordem de lançamento —
isso garante que cada passo termine antes do próximo iniciar, sem necessidade de
`cudaDeviceSynchronize()` dentro do kernel pai (que além de inútil aqui, está depreciado no
CDP2 e gera warning de compilação). A estrutura é deliberadamente idêntica à versão CUDA flat
em tudo exceto onde o loop de controle roda (device vs host) — isola o efeito do overhead de
lançamento de kernel a partir do device como variável de comparação.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_XX -rdc=true -o bitonic_cudadp bitonic_cudadp.cu
```

**Como executar**:
```bash
./bitonic_cudadp N [iteracoes]
```

**Saída esperada**:
```
bitonic_sort|cudadp|aleatorio|1048576|1|0,003342|1|256x2048_dp
```

## Notas importantes

- Execução e validação de corretude na RTX 4090 (CUDA e CUDA DP) ainda pendentes — os quatro
  binários compilam e foram exercitados localmente, mas os resultados de tempo e corretude no
  hardware alvo do projeto ainda não foram coletados.
- Nota honesta sobre CUDA DP: para Bitonic Sort, a estrutura de execução é regular e estática —
  todos os passos têm o mesmo formato de lançamento (`N/2` threads, sem irregularidade entre
  blocos). Isso não joga a favor do CUDA DP, cujo benefício típico vem de adaptar o paralelismo
  a estruturas irregulares ou recursivas com tamanhos de subproblema variáveis. O overhead de
  lançamento de kernel a partir do device tende a não ser compensado por ganho algorítmico
  aqui — espera-se desempenho igual ou inferior ao CUDA flat, e esse resultado, se confirmado,
  é o esperado e não indica erro de implementação.
- CDP2 vs CDP1: este projeto usa CUDA 12.1, onde CDP2 é o único modelo disponível (CDP1 foi
  removido a partir do CUDA 12.0). A principal implicação prática no código é que
  `cudaDeviceSynchronize()` chamado a partir do device está depreciado e não deve ser usado para
  sincronizar kernels filhos — a serialização entre lançamentos sucessivos na stream default
  por-thread já é suficiente para a dependência de ordem exigida pelo Bitonic Sort.

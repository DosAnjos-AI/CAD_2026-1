# Floyd-Warshall

## 1. Visão geral do algoritmo

Floyd-Warshall calcula o caminho mínimo entre todos os pares de vértices de um
grafo ponderado. Complexidade de tempo O(V³) — três loops aninhados sobre
vértice intermediário (`k`), origem (`i`) e destino (`j`). Complexidade de
espaço O(V²), já que o estado é a própria matriz de distâncias `dist[V][V]`,
atualizada in-place a cada iteração de `k`.

A matriz é densa, alocada como vetor 1D `int32_t` de tamanho V*V em row-major
(`dist[i*V+j]`), favorecendo coalescência de acesso na GPU. A geração é
determinística via `srand(42)`: diagonal principal zerada, e para cada par
`i != j` uma moeda (`rand() % 2`) decide se há aresta — em caso positivo o
peso é `(rand() % 1000) + 1`, caso contrário `INF` (1e9). Essa geração produz
densidade de arestas próxima de 50%, como definido em `decisoes_tecnicas.md`.

## 2. Por implementação

### CPU sequencial — `floyd_cpu.c`

**O que faz**: triplo loop clássico `k → i → j`, atualizando
`dist[i*V+j] = min(dist[i*V+j], dist[i*V+k] + dist[k*V+j])`.

**Decisão técnica relevante**: poda dupla com `if (dist[i*V+k] < INF)` antes
do loop de `j` e `if (dist[k*V+j] < INF)` dentro dele. Evita somar dois `INF`
(o que causaria overflow de `int32_t`, já que `INF + INF` excede o range) e
também economiza trabalho em pares sem caminho via `k`. Serve como oráculo de
corretude para as demais APIs.

**Como compilar**:
```bash
gcc -O3 -march=native -o floyd_cpu floyd_cpu.c
```

**Como executar**:
```bash
./floyd_cpu V [iteracoes]
```

**Saída esperada**:
```
floyd_warshall|cpu|aleatorio|1024|1|0,823451|1|1x1
```

### OpenMP — `floyd_openmp.c`

**O que faz**: mesma estrutura de três loops, mas o loop `k` permanece
sequencial no host (laço externo) enquanto o loop `i` é paralelizado com
`#pragma omp parallel for private(j)`; o loop `j` roda sequencial dentro de
cada thread.

**Decisão técnica relevante**: o loop `k` não pode ser paralelizado pois cada
iteração depende do resultado da anterior (a matriz inteira é atualizada
antes de avançar para o próximo vértice intermediário). A variável `int j` é
declarada antes do `#pragma omp parallel for` para permitir `private(j)`
válido em C — a cláusula `private` exige que a variável já exista no escopo
externo ao pragma; declará-la dentro do `for` quebraria a compilação com essa
cláusula.

**Como compilar**:
```bash
gcc -O3 -march=native -fopenmp -DNUM_THREADS=$(nproc) -o floyd_openmp floyd_openmp.c
```

**Como executar**:
```bash
./floyd_openmp V [iteracoes]
```

**Saída esperada**:
```
floyd_warshall|openmp|aleatorio|1024|1|0,051203|1|32x1
```

### CUDA — `floyd_cuda.cu`

**O que faz**: kernel `fw_kernel` com um thread por par `(i,j)`, lançado V
vezes pelo host — uma vez por valor de `k`. Grid 2D: `dim3 blocks(nblocos,
nblocos)` com `nblocos = ceil(V/16)`, `dim3 threads(16,16)`.

**Decisão técnica relevante**: `BLOCK_SIZE = 16` para formar blocos de 256
threads (16×16), adequado a um kernel 2D. Implementação em memória global
pura, sem uso de shared memory — opção deliberada por KISS, em linha com as
demais implementações CUDA do projeto. A sincronização entre iterações de `k`
é garantida implicitamente pelo lançamento sequencial de kernels no mesmo
stream (cada `fw_kernel<<<...>>>` só inicia após o anterior terminar).

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -o floyd_cuda floyd_cuda.cu
```

**Como executar**:
```bash
./floyd_cuda V [iteracoes]
```

**Saída esperada**:
```
floyd_warshall|cuda|aleatorio|1024|1|0,008912|1|16x4096
```

### CUDA DP — `floyd_cudadp.cu`

**O que faz**: o kernel filho `fw_kernel` é idêntico ao de `floyd_cuda.cu`.
O controle do loop de `k` é movido para o device através do kernel pai
`fw_dp_launcher<<<1,1>>>`, que lança `fw_kernel` para a iteração atual e, em
seguida, relança a si mesmo para `k+1` via `cudaStreamTailLaunch`.

**Decisão técnica relevante**: `cudaStreamTailLaunch` garante que o
relançamento de `fw_dp_launcher(k+1)` só comece depois que `fw_kernel(k)`
termine — é a mesma técnica usada em `merge_cudadp.cu` para serializar
estágios dependentes sem exigir sincronização explícita do host entre cada
iteração. Essa é uma divergência da abordagem R-Kleene (divisão recursiva em
quadrantes) descrita em `decisoes_tecnicas.md`: optou-se por uma recursão
sequencial de profundidade V no device, mantendo o mesmo kernel de
atualização do CUDA flat, em vez de decompor a matriz em submatrizes
independentes. A justificativa é a mesma adotada no merge sort — reaproveitar
um padrão já validado de serialização via DP, mantendo KISS.

**Nota honesta**: como a recursão apenas substitui o loop do host por um loop
de profundidade V no device — sem paralelismo adicional entre quadrantes —
não se espera ganho sobre o CUDA flat. Pelo contrário: cada nível de recursão
adiciona overhead de lançamento de kernel filho (latência de child launch é
maior que lançamento do host) e ainda há a sobrecarga acumulada de V
lançamentos de `fw_dp_launcher`. O resultado esperado é CUDA DP mais lento que
CUDA flat para Floyd-Warshall, e isso deve ser documentado como resultado
válido — não como falha de implementação.

**Como compilar**:
```bash
nvcc -O3 -arch=sm_89 -rdc=true -o floyd_cudadp floyd_cudadp.cu
```

**Como executar**:
```bash
./floyd_cudadp V [iteracoes]
```

**Saída esperada**:
```
floyd_warshall|cudadp|aleatorio|1024|1|0,015734|1|1x1_dp
```

## 3. Notas importantes

- **Ordem de `rand()` em `gerar_matriz`**: todas as quatro implementações
  percorrem `i` de 0 a V-1 e, para cada `i`, `j` de 0 a V-1, chamando
  `rand() % 2` e condicionalmente `rand() % 1000` na mesma ordem. Isso é
  obrigatório porque `rand()` é um gerador determinístico baseado em estado
  global: qualquer mudança na ordem ou quantidade de chamadas produz uma
  matriz diferente, invalidando a comparação de corretude contra o oráculo
  sequencial.
- **OpenMP — `private(j)`**: ver decisão técnica na seção do OpenMP acima.
- **CUDA — memória global pura**: sem shared memory, por simplicidade (KISS),
  igual às demais implementações CUDA do projeto.
- **CUDA DP — `cudaStreamTailLaunch`**: serializa as V iterações de `k` no
  device, mesma solução usada em `merge_cudadp.cu` para encadear estágios
  dependentes sem sincronização do host.
- **Overhead esperado do CUDA DP**: ver nota honesta na seção do CUDA DP
  acima — espera-se desempenho inferior ao CUDA flat.
- **Pendências de validação na RTX 4090**: os binários foram compilados e
  testados no ambiente de desenvolvimento atual; a validação de corretude e
  coleta de tempos no hardware alvo (RTX 4090, sm_89) listado no
  `CLAUDE.md` ainda não foi executada.

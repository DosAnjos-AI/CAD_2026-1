# HOWTO — Guia prático de uso

Guia de execução do benchmark CAD_2026_v3. Para detalhes de implementação de
cada algoritmo, ver os `README.md` dentro de `bitonic_sort/`, `merge_sort/`,
`bfs/` e `floyd_warshall/`. Para decisões técnicas de projeto, ver
`decisoes_tecnicas.md` e `CLAUDE.md`.

## 1. Pré-requisitos

- `gcc-12` (ou compatível) com suporte a `-march=native` e `-fopenmp`
- `nvcc` (CUDA Toolkit) — versão alvo do projeto: CUDA 12.1.105
- GPU NVIDIA com suporte a CUDA Dynamic Parallelism (Compute Capability >= 3.5)
  e `cudaStreamTailLaunch` (CDP2, a partir do CUDA 11.6) — hardware alvo:
  RTX 4090, `sm_89`
- `nvidia-smi` disponível no PATH (usado pelo `run_all.sh` para detectar a
  arquitetura da GPU)
- `nproc` disponível (padrão em qualquer Linux) — usado para detectar o
  número de threads da CPU

Verificar antes de rodar:

```bash
gcc --version
nvcc --version
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
nproc
```

## 2. Como executar o benchmark completo

Comando único, em background, sobrevivendo a desconexão de terminal:

```bash
nohup ./run_all.sh > results/log.txt 2>&1 &
```

O script não aceita argumentos — só roda como `./run_all.sh`. Ele compila
todos os binários, cria `results/resultados.csv` se necessário e processa
todas as combinações `algoritmo|api|tamanho` na ordem definida em
`CLAUDE.md`.

Acompanhar progresso em tempo real:

```bash
tail -f results/log.txt
```

Cada linha de log mostra horário, combinação em execução e skips de
combinações já completas. Para ver o CSV crescendo:

```bash
watch -n 5 wc -l results/resultados.csv
```

Estimativa de tempo total: ~48h para rodar as 4 etapas de algoritmo × 4 APIs
× 9 (ou 8) tamanhos completos no hardware alvo (RTX 4090 / i9-14900K). Tempo
real varia com a carga térmica da máquina e com os tamanhos maiores
(2^26 para vetores, 2^24 para BFS, 2^13 para Floyd-Warshall).

## 3. Como retomar após interrupção

Se a máquina desligar ou o processo for interrompido no meio da execução:

- O `results/resultados.csv` mantém todas as linhas já gravadas até o ponto
  da interrupção — nada é perdido.
- A combinação que estava em andamento no momento da interrupção fica
  parcial (menos de 10 linhas medidas).
- Basta rodar `nohup ./run_all.sh > results/log.txt 2>&1 &` novamente. O
  script:
  1. Recompila todos os binários (idempotente, sem custo relevante)
  2. Para cada combinação, verifica `combinacao_completa()`: se já tem 10
     linhas no CSV, pula (incluindo warmup)
  3. Se a combinação está parcial, `limpar_parcial()` remove as linhas
     incompletas dessa combinação e ela é reprocessada do zero (3 warmups +
     10 medidas)

Verificar o estado atual do CSV antes de retomar:

```bash
cut -d'|' -f1,2,4 results/resultados.csv | sort | uniq -c
```

Combinações com contagem >= 10 estão completas; o restante será
reprocessado na próxima execução.

## 4. Como executar um algoritmo individual

Todos os binários aceitam o tamanho como argumento obrigatório e um número
de iterações opcional (útil para teste rápido sem rodar as 10 iterações
medidas completas):

```bash
./bitonic_sort/bitonic_cpu 1048576 1
./bitonic_sort/bitonic_openmp 1048576 1
./bitonic_sort/bitonic_cuda 1048576 1
./bitonic_sort/bitonic_cudadp 1048576 1
```

Mesma sintaxe para `merge_sort/`, `bfs/` e `floyd_warshall/` (trocando
apenas o nome do binário e o significado do tamanho — N para vetores, V
para grafos). Exemplo com Floyd-Warshall:

```bash
./floyd_warshall/floyd_cuda 1024 1
```

Sem o segundo argumento, o binário roda o protocolo completo (3 warmup +
10 medidas internas por iteração). A saída vai para `stdout` no formato do
CSV — para gravar manualmente:

```bash
./bfs/bfs_cpu 65536 >> results/resultados.csv
```

Os binários não compilados precisam ser gerados antes (ver flags exatas em
`CLAUDE.md`, seção "Flags de compilação", ou rodar `./run_all.sh` uma vez
para compilar todos de uma vez).

## 5. Formato do CSV de saída

Arquivo: `results/resultados.csv`

```
sep=|
algoritmo|api|cenario|tamanho|iteracao|tempo_s|corretude|threads_blocos
```

Campos:

| Campo | Descrição |
|---|---|
| `algoritmo` | `bitonic_sort`, `merge_sort`, `bfs` ou `floyd_warshall` |
| `api` | `cpu`, `openmp`, `cuda` ou `cudadp` |
| `cenario` | sempre `aleatorio` neste projeto |
| `tamanho` | N (vetores/BFS) ou V (Floyd-Warshall), inteiro |
| `iteracao` | número da iteração medida (1 a 10) |
| `tempo_s` | tempo médio das 10 execuções internas da iteração, decimal com vírgula |
| `corretude` | `1` se idêntico ao oráculo sequencial, `0` caso contrário |
| `threads_blocos` | configuração de paralelismo usada (ver formato por API/algoritmo nos READMEs) |

Exemplo de linhas reais:

```
bitonic_sort|cpu|aleatorio|1048576|1|0,823451|1|1x1
bitonic_sort|openmp|aleatorio|1048576|1|0,312045|1|32x1
bitonic_sort|cuda|aleatorio|1048576|1|0,002981|1|256x2048
bfs|cudadp|aleatorio|1048576|1|0,015234|1|1x1_dp
floyd_warshall|cuda|aleatorio|1024|1|0,008912|1|16x4096
```

Como abrir no Excel (locale brasileiro): a primeira linha `sep=|` já
instrui o Excel a usar `|` como separador automaticamente ao abrir o
arquivo (duplo clique ou "Abrir"). Os decimais com vírgula são lidos
corretamente sem configuração adicional, desde que o Windows/Excel esteja
configurado para o locale `pt-BR`. Se abrir como texto puro (sem
reconhecer separador), usar Dados > Texto para Colunas, delimitador `|`.

## 6. Estrutura do projeto

```
CAD_2026-1/
├── run_all.sh              orquestrador: compila e executa tudo
├── CLAUDE.md                regras e definições do projeto
├── decisoes_tecnicas.md     decisões de implementação por algoritmo
├── HOWTO.md                 este guia
├── results/
│   ├── resultados.csv       CSV append-only com todos os resultados
│   └── log.txt              log gerado por nohup ao rodar run_all.sh
├── bitonic_sort/
│   ├── README.md
│   ├── bitonic_cpu.c / bitonic_openmp.c / bitonic_cuda.cu / bitonic_cudadp.cu
│   └── bitonic_cpu / bitonic_openmp / bitonic_cuda / bitonic_cudadp  (binários compilados)
├── merge_sort/        (mesma estrutura, prefixo merge_)
├── bfs/                (mesma estrutura, prefixo bfs_)
└── floyd_warshall/     (mesma estrutura, prefixo floyd_)
```

Cada binário compilado fica dentro do diretório do próprio algoritmo, junto
do código-fonte. `results/` é criado automaticamente pelo `run_all.sh` se
não existir.

## 7. Solução de problemas comuns

**GPU não detectada (`nvidia-smi` falha ou retorna vazio)**
`run_all.sh` usa `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
para definir `ARCH=sm_XX`. Se isso falhar, a compilação dos binários CUDA
quebra com `ARCH` vazio. Verificar driver instalado com `nvidia-smi` puro
antes de rodar o script; se a GPU não for detectada, os binários CUDA e
CUDA DP não serão compilados, mas CPU e OpenMP continuam funcionando
normalmente (o script não usa `set -e`).

**Compilação falhou para um binário específico**
O log mostra `[ERRO] Falha ao compilar <nome>`. O script continua mesmo com
falha de compilação — combinações que dependem desse binário vão falhar em
`executar()` com `[ERRO] Binario nao encontrado` e serão puladas. Compilar
manualmente o binário problemático isoladamente para ver o erro completo,
por exemplo:

```bash
nvcc -O3 -arch=sm_89 -rdc=true -o bfs/bfs_cudadp bfs/bfs_cudadp.cu
```

**CSV com linhas duplicadas ou corrompidas**
O CSV é append-only — duplicação geralmente vem de execução manual repetida
do mesmo binário redirecionando para o CSV (seção 4) sem passar pelo
controle de skip do `run_all.sh`. Remover linhas manualmente com `grep -v`
filtrando pela combinação afetada, mantendo as linhas `sep=|` e o
cabeçalho:

```bash
grep -v "^bitonic_sort|cpu|aleatorio|1048576|" results/resultados.csv > /tmp/csv_limpo
mv /tmp/csv_limpo results/resultados.csv
```

**Forçar reprocessamento de uma combinação específica**
Apagar as linhas dessa combinação do CSV antes de rodar `run_all.sh` de
novo — com menos de 10 linhas restantes, `combinacao_completa()` retorna
falso e a combinação é reprocessada do zero (warmup + 10 medidas):

```bash
grep -v "^merge_sort|cuda|aleatorio|4194304|" results/resultados.csv > /tmp/csv_limpo
mv /tmp/csv_limpo results/resultados.csv
nohup ./run_all.sh > results/log.txt 2>&1 &
```

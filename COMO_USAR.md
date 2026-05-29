# Como Usar

## Pré-requisitos

- `gcc` com suporte a OpenMP (`-fopenmp`)
- CUDA Toolkit compatível com a GPU local
- `nvidia-smi` (GPUs discretas) ou `tegrastats` (Jetson AGX Orin)
- `make`

## Clonar o repositório

```bash
git clone https://github.com/DosAnjos-AI/CAD_2026-1
cd CAD_2026-1
```

## Configurar o hardware

Editar as variáveis no início do script escolhido:

```bash
HARDWARE="mx350"   # opções: mx350 | rtx4090 | jetson_agx_orin
ITERACOES=10       # número de repetições externas (linhas no CSV)
RUNS=10000         # execuções internas por iteração
```

## Executar o benchmark completo

**Sem sudo** (energia CPU registrada como N/A):
```bash
bash scripts/run_benchmarks.sh
```

**Com sudo** (coleta energia CPU via `perf stat`):
```bash
sudo bash scripts/run_benchmarks_sudo.sh
```

Ambos os scripts compilam todos os 12 binários, verificam os executáveis
e iniciam as execuções automaticamente.

## Estrutura do Benchmark

### Warm-up

Cada binário executa **1 rodada de aquecimento** antes de medir o tempo.
O warm-up elimina o overhead de inicialização do contexto CUDA, alocação
de memória na GPU e cold-start de caches. Ele também é a execução usada
para verificar a **corretude** do resultado.

### Loop interno (--runs N)

Após o warm-up, cada binário executa o algoritmo **N vezes consecutivas**
dentro de um único intervalo `gettimeofday()`:

```
gettimeofday(&inicio)
  for r in [0..N):
    reinicializa dados (vetor embaralhado, distâncias zeradas, etc.)
    executa o algoritmo
gettimeofday(&fim)
```

O campo `tempo_total_s` no CSV é o tempo das **N execuções juntas**,
não o tempo de uma única execução.

Para obter o tempo médio por execução: `tempo_total_s / runs`.

### Iterações externas (ITERACOES)

O script externo repete cada configuração `ITERACOES` vezes, gerando
múltiplas linhas no CSV. Isso permite calcular médias e desvios padrão
entre as iterações.

## Onde ficam os resultados

```
results/<hardware>/
  mergesort_openmp.csv
  mergesort_cuda.csv
  mergesort_cuda_dp.csv
  quicksort_openmp.csv
  ...
```

Formato de cada linha (9 campos):

```
algoritmo,api,hardware,tamanho,iteracao,tempo_total_s,energia_gpu_j,energia_cpu_j,corretude
mergesort,openmp,mx350,100000,1,55.302000,N/A,12.45,OK
```

| Campo | Descrição |
|-------|-----------|
| `tempo_total_s` | Tempo total das `runs` execuções internas (segundos) |
| `energia_gpu_j` | Energia GPU estimada (nvidia-smi média × tempo) |
| `energia_cpu_j` | Energia CPU (perf energy-pkg, apenas run_benchmarks_sudo.sh) |
| `corretude` | OK ou ERRO — verificado no warm-up |

## Configuração para coleta de energia CPU

O `run_benchmarks_sudo.sh` usa `sudo perf stat -e power/energy-pkg/` para
medir a energia da CPU. Em execuções em background (ex: `nohup`), o sudo
suspende o processo ao pedir senha interativamente.

Para evitar isso, configure o sudoers para permitir `perf` sem senha.
Execute **uma vez em cada máquina**:

```bash
sudo visudo
```

Adicione a linha abaixo (substitua `SEU_USUARIO` pelo usuário da máquina):

```
SEU_USUARIO ALL=(ALL) NOPASSWD: /usr/bin/perf
```

Salve e feche. Para verificar se a configuração está correta:

```bash
sudo -n perf stat -e power/energy-pkg/ sleep 1
```

Se não retornar erro de senha, está configurado corretamente.

> **Sem essa configuração:** use `run_benchmarks.sh` (Plano B).
> O tempo e a energia GPU são coletados normalmente; apenas
> `energia_cpu_j` ficará como `N/A`.

## Diferença entre os dois scripts

| Aspecto | `run_benchmarks.sh` | `run_benchmarks_sudo.sh` |
|---------|--------------------|-----------------------|
| Requer sudo | Não | Sim |
| Energia CPU | N/A | `perf stat -e power/energy-pkg/` |
| Energia GPU | nvidia-smi (GPUs discretas) | idem |
| Jetson tegrastats | sem sudo | com sudo |

## Observações

- **Execuções rápidas (< 1s):** energia GPU pode ser N/A — resolução
  do `nvidia-smi` é de 1 segundo por amostra.
- **Interrupção:** `Ctrl+C` encerra o script. Arquivos CSV gerados até
  o momento são mantidos.
- **Teste rápido:** `TEST_MODE=1 RUNS=5 bash scripts/run_benchmarks.sh`
  executa apenas mergesort/openmp/N=100 com 1 iteração para verificar
  o ambiente sem rodar o benchmark completo.

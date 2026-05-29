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

Editar a linha 10 de `scripts/run_benchmarks.sh`:

```bash
HARDWARE="mx350"   # opções: mx350 | rtx4090 | jetson_agx_orin
```

## Executar o benchmark completo

```bash
bash scripts/run_benchmarks.sh
```

O script compila todos os 12 binários, verifica os executáveis e
inicia as execuções automaticamente.

## Onde ficam os resultados

```
results/<hardware>/
  mergesort_openmp.csv
  mergesort_cuda.csv
  mergesort_cuda_dp.csv
  quicksort_openmp.csv
  ...
```

Formato de cada linha:

```
algoritmo,api,hardware,tamanho,iteracao,tempo_s,energia_j,corretude
mergesort,openmp,mx350,100000,1,0.055302,12.45,OK
```

## Observações

- **Jetson:** `tegrastats` pode exigir `sudo`. Se não estiver disponível,
  energia é registrada como `N/A`.
- **Execuções rápidas (< 1s):** energia registrada como `N/A` — resolução
  do `nvidia-smi` é de 1 segundo.
- **Interrupção:** `Ctrl+C` encerra o script. Arquivos CSV gerados até
  o momento são mantidos.

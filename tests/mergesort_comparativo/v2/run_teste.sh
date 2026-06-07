#!/bin/bash

PASTA_SRC="$(dirname "$0")/src"
PASTA_OUT="$(dirname "$0")/outputs"
mkdir -p "$PASTA_OUT"

RESULTADO="$PASTA_OUT/resultados.csv"

# flags de compilacao CUDA
NVCC=/usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS="-O2 -rdc=true -lcudadevrt \
           -ccbin gcc-12 \
           -arch=compute_61 -code=sm_61 \
           -DCUDA_FORCE_CDP1_IF_SUPPORTED \
           -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
           -Xcompiler -Wno-deprecated-declarations"

SLEEP_ENTRE_VERSOES=20   # segundos entre sequencial/artigo/otimizado
SLEEP_ENTRE_TAMANHOS=30  # segundos entre tamanhos diferentes

# compilacao
echo "Compilando mergesort_sequencial..."
gcc -O2 -o "$PASTA_SRC/mergesort_sequencial" "$PASTA_SRC/mergesort_sequencial.c"

echo "Compilando mergesort_artigo..."
$NVCC $NVCCFLAGS -o "$PASTA_SRC/mergesort_artigo" "$PASTA_SRC/mergesort_artigo.cu"

echo "Compilando mergesort_otimizado..."
$NVCC $NVCCFLAGS -o "$PASTA_SRC/mergesort_otimizado" "$PASTA_SRC/mergesort_otimizado.cu"

# cabecalho CSV (unica vez)
echo "algoritmo,versao,n,iteracao,execucoes,tempo_total_s,energia_cpu_j,energia_gpu_j,corretude" \
    > "$RESULTADO"

# execucao por tamanho
TAMANHOS=(100 10000 100000 1000000 8000000)
ULTIMO_IDX=$(( ${#TAMANHOS[@]} - 1 ))

for i in "${!TAMANHOS[@]}"; do
    N=${TAMANHOS[$i]}
    echo "=== Tamanho N=$N ==="

    echo "  Executando sequencial n=$N..."
    sudo "$PASTA_SRC/mergesort_sequencial" $N >> "$RESULTADO"
    sleep $SLEEP_ENTRE_VERSOES

    echo "  Executando artigo n=$N..."
    sudo "$PASTA_SRC/mergesort_artigo" $N >> "$RESULTADO"
    sleep $SLEEP_ENTRE_VERSOES

    echo "  Executando otimizado n=$N..."
    sudo "$PASTA_SRC/mergesort_otimizado" $N >> "$RESULTADO"

    # sleep entre tamanhos (exceto apos o ultimo)
    if [ $i -lt $ULTIMO_IDX ]; then
        echo "  Aguardando $SLEEP_ENTRE_TAMANHOS s antes do proximo tamanho..."
        sleep $SLEEP_ENTRE_TAMANHOS
    fi
done

echo "Concluido. Resultados em $RESULTADO"

#!/bin/bash

NVCC=/usr/local/cuda-12.2/bin/nvcc
NVCCFLAGS="-O2 -rdc=true -lcudadevrt \
     -ccbin gcc-12 \
     -arch=compute_61 -code=sm_61 \
     -DCUDA_FORCE_CDP1_IF_SUPPORTED \
     -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING \
     -Xcompiler -Wno-deprecated-declarations"

PASTA_SRC="$(dirname "$0")/src"
PASTA_OUT="$(dirname "$0")/outputs"
mkdir -p "$PASTA_OUT"

# compilacao
echo "Compilando mergesort_artigo..."
$NVCC $NVCCFLAGS \
     -o "$PASTA_SRC/mergesort_artigo" \
     "$PASTA_SRC/mergesort_artigo.cu"

echo "Compilando mergesort_otimizado..."
$NVCC $NVCCFLAGS \
     -o "$PASTA_SRC/mergesort_otimizado" \
     "$PASTA_SRC/mergesort_otimizado.cu"

# execucao para cada tamanho
for N in 100 10000 100000 1000000; do
    echo "Executando artigo    n=$N..."
    "$PASTA_SRC/mergesort_artigo" $N \
        >> "$PASTA_OUT/resultados_artigo.csv" 2>&1

    echo "Executando otimizado n=$N..."
    "$PASTA_SRC/mergesort_otimizado" $N \
        >> "$PASTA_OUT/resultados_otimizado.csv" 2>&1
done

echo "Concluido. Resultados em $PASTA_OUT/"

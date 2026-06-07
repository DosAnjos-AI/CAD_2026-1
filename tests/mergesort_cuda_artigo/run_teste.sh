#!/bin/bash
# Benchmark mergesort_cuda_artigo — versão fiel ao artigo Nogueira et al. 2024
# Energia CPU/GPU fixadas em 0 (MX350 sem suporte a leitura de energia)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/src/mergesort_cuda_artigo"
OUT_DIR="$SCRIPT_DIR/outputs"
CSV="$OUT_DIR/resultados.csv"
LOG="$OUT_DIR/log.txt"

SIZES=(100 10000 100000 1000000)
ITERACOES=5
EXECUCOES=10

mkdir -p "$OUT_DIR"

# Cabeçalho CSV escrito uma única vez
echo "algoritmo,versao,n,iteracao,execucoes,tempo_total_s,energia_cpu_j,energia_gpu_j,corretude" > "$CSV"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark iniciado" | tee -a "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Tamanhos: ${SIZES[*]} | Iteracoes: $ITERACOES | Execucoes/iter: $EXECUCOES" | tee -a "$LOG"

for i_n in "${!SIZES[@]}"; do
    n="${SIZES[$i_n]}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Tamanho n=$n ===" | tee -a "$LOG"

    for iter in $(seq 1 "$ITERACOES"); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] n=$n iteracao=$iter/$ITERACOES" | tee -a "$LOG"

        saida=$("$BIN" "$n" --runs "$EXECUCOES" 2>>"$LOG")
        status=$?

        if [ "$status" -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERRO: binario retornou status $status para n=$n iter=$iter" | tee -a "$LOG"
            continue
        fi

        energia_cpu="0.000000"

        # Saida do binario: mergesort,cuda,N,RUNS,TEMPO,CORRETUDE
        tempo=$(echo "$saida" | cut -d',' -f5)
        corretude=$(echo "$saida" | cut -d',' -f6)

        echo "mergesort,cuda,$n,$iter,$EXECUCOES,$tempo,$energia_cpu,0.000000,$corretude" >> "$CSV"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] n=$n iter=$iter tempo=${tempo}s energia_cpu=${energia_cpu}J corretude=$corretude" | tee -a "$LOG"

        # 20s entre iteracoes do mesmo tamanho (exceto na ultima)
        if [ "$iter" -lt "$ITERACOES" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Aguardando 20s entre iteracoes..." | tee -a "$LOG"
            sleep 20
        fi
    done

    # 30s entre tamanhos diferentes (exceto no ultimo)
    if [ "$i_n" -lt $(( ${#SIZES[@]} - 1 )) ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Aguardando 30s entre tamanhos..." | tee -a "$LOG"
        sleep 30
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark concluido" | tee -a "$LOG"

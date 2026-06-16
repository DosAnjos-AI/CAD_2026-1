#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define INF 1000000000
#define BLOCK_SIZE 16

/* Um thread por par (i, j): atualiza dist[i][j] usando o vertice intermediario k */
__global__ void fw_kernel(int32_t *dist, int v, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= v || j >= v) return;

    int d_ik = dist[i * v + k];
    int d_kj = dist[k * v + j];
    if (d_ik < INF && d_kj < INF) {
        int novo = d_ik + d_kj;
        if (novo < dist[i * v + j])
            dist[i * v + j] = novo;
    }
}

/* Gera matriz de adjacencia densa V x V, row-major, com semente fixa */
static void gerar_matriz(int32_t *dist, int v) {
    srand(42);
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < v; j++) {
            if (i == j) {
                dist[i * v + j] = 0;
            } else if ((rand() % 2) == 0) {
                dist[i * v + j] = (rand() % 1000) + 1;
            } else {
                dist[i * v + j] = INF;
            }
        }
    }
}

/* Floyd-Warshall sequencial puro, usado apenas como oraculo de corretude */
static void floyd_warshall_seq(int32_t *dist, int v) {
    for (int k = 0; k < v; k++)
        for (int i = 0; i < v; i++)
            if (dist[i * v + k] < INF)
                for (int j = 0; j < v; j++)
                    if (dist[k * v + j] < INF)
                        if (dist[i * v + k] + dist[k * v + j] < dist[i * v + j])
                            dist[i * v + j] = dist[i * v + k] + dist[k * v + j];
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Uso: %s <V> [iteracoes]\n", argv[0]);
        return 1;
    }

    int v = atoi(argv[1]);
    int iteracoes = (argc == 3) ? atoi(argv[2]) : 10;

    if (v <= 0) {
        fprintf(stderr, "Erro: V deve ser positivo (recebido: %d)\n", v);
        return 1;
    }

    if (iteracoes <= 0) {
        fprintf(stderr, "Erro: iteracoes deve ser positivo (recebido: %d)\n", iteracoes);
        return 1;
    }

    int nblocos = (v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(nblocos, nblocos);

    int32_t *dist = (int32_t *)malloc((size_t)v * (size_t)v * sizeof(int32_t));
    int32_t *dist_ref = (int32_t *)malloc((size_t)v * (size_t)v * sizeof(int32_t));
    if (!dist || !dist_ref) {
        fprintf(stderr, "Erro: falha ao alocar memoria\n");
        free(dist);
        free(dist_ref);
        return 1;
    }

    int32_t *d_dist;
    if (cudaMalloc(&d_dist, (size_t)v * (size_t)v * sizeof(int32_t)) != cudaSuccess) {
        fprintf(stderr, "Erro: falha ao alocar memoria na GPU\n");
        free(dist);
        free(dist_ref);
        return 1;
    }

    /* Oraculo: executa uma vez antes do loop principal */
    gerar_matriz(dist_ref, v);
    floyd_warshall_seq(dist_ref, v);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* 3 execucoes de warmup, sem saida */
    for (int w = 0; w < 3; w++) {
        gerar_matriz(dist, v);
        cudaMemcpy(d_dist, dist, (size_t)v * (size_t)v * sizeof(int32_t), cudaMemcpyHostToDevice);
        for (int k = 0; k < v; k++)
            fw_kernel<<<blocks, threads>>>(d_dist, v, k);
        cudaDeviceSynchronize();
    }

    for (int iter = 1; iter <= iteracoes; iter++) {
        double soma = 0.0;
        int corretude = 1;

        for (int exec = 0; exec < 10; exec++) {
            gerar_matriz(dist, v);
            cudaMemcpy(d_dist, dist, (size_t)v * (size_t)v * sizeof(int32_t), cudaMemcpyHostToDevice);

            cudaEventRecord(start);
            for (int k = 0; k < v; k++)
                fw_kernel<<<blocks, threads>>>(d_dist, v, k);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            soma += (double)ms / 1000.0;

            cudaMemcpy(dist, d_dist, (size_t)v * (size_t)v * sizeof(int32_t), cudaMemcpyDeviceToHost);

            if (iter == 1) {
                for (int i = 0; i < v * v; i++) {
                    if (dist[i] != dist_ref[i]) {
                        corretude = 0;
                        break;
                    }
                }
            }
        }

        double tempo_s = soma / 10.0;

        char tempo_str[64];
        snprintf(tempo_str, sizeof(tempo_str), "%.6f", tempo_s);
        for (int i = 0; tempo_str[i] != '\0'; i++) {
            if (tempo_str[i] == '.') {
                tempo_str[i] = ',';
                break;
            }
        }

        printf("floyd_warshall|cuda|aleatorio|%d|%d|%s|%d|16x%d\n",
               v, iter, tempo_str, corretude, nblocos);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_dist);
    free(dist);
    free(dist_ref);
    return 0;
}

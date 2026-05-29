# Replicação e Extensão — Nogueira et al. (SSCAD 2024)

Replicação e extensão dos experimentos do artigo *"Análise de Desempenho e Consumo Energético de Aplicações Recursivas em Ambientes OpenMP, CUDA e CUDA DP"* (Nogueira et al., SSCAD 2024).

## Sobre o artigo original

Avalia 4 algoritmos recursivos (Mergesort, Quicksort, BFS e SSSP) nas APIs OpenMP, CUDA e CUDA DP, medindo tempo de execução e consumo de energia (CPU + GPU) em uma máquina com AMD Ryzen 9 3900X e GeForce GTX 1050.

## O que este repositório adiciona

- Replicação dos benchmarks originais nos mesmos algoritmos e APIs
- Execução em novos ambientes de hardware: **GTX 1650**, **Jetson AGX Orin** e **RTX 4090**
- Análise estatística aprimorada em relação ao artigo original

## Algoritmos

| Algoritmo | Tipo |
|-----------|------|
| Mergesort | Ordenação vetorial |
| Quicksort | Ordenação vetorial |
| BFS | Busca em grafos |
| SSSP | Busca em grafos |

## Hardware avaliado

| Hardware | Contexto |
|----------|----------|
| GTX 1650 | Desktop GPU entrada |
| Jetson AGX Orin | Edge AI SoM |
| RTX 4090 | Desktop GPU topo |
| MX350 | Desenvolvimento e validação local |

## Configuração do Ambiente de Desenvolvimento

Cada colaborador deve configurar seu próprio ambiente local seguindo
as convenções do projeto:

- Compilador C com suporte a OpenMP (gcc com flag -fopenmp)
- CUDA Toolkit compatível com a GPU local
- Host compiler compatível com a versão do CUDA Toolkit utilizada
- make para automação de build
- Seguir o fluxo Git do projeto: feature/xxx -> main

Consulte os Makefiles em cada diretório de algoritmo/API para os
comandos de compilação e execução específicos.

## Formato de Saída e Coleta de Resultados

Cada binário imprime uma linha CSV por execução no stdout:

```
algoritmo,api,tamanho,tempo_s,corretude
mergesort,openmp,100000,0.079210,OK
```

Os scripts de benchmark redirecionam a saída em append para arquivos
CSV em `results/<hardware>/`, um arquivo por combinação algoritmo/API:

```
results/
  mx350/
    mergesort_openmp.csv
    mergesort_cuda.csv
    mergesort_cuda_dp.csv
    ...
```

Cada CSV acumula uma linha por execução. Com 10 iterações por tamanho,
o arquivo final terá 30 linhas (3 tamanhos × 10 iterações) para
algoritmos de ordenação, e 40 linhas (4 tamanhos × 10 iterações)
para SSSP.

## Status

> Em desenvolvimento — scripts ainda não implementados.

## Referência

Nogueira, A. G. D., Lorenzon, A. F., Schepke, C., Kreutz, D. *Análise de Desempenho e Consumo Energético de Aplicações Recursivas em Ambientes OpenMP, CUDA e CUDA DP*. SSCAD 2024, São Carlos/SP, pp. 264–275.

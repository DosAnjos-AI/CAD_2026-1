import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = "results/rtx4090/CAD_rtx4090.csv"
OUT_DIR = "results/graficos"

CORES  = {"openmp": "#4472C4", "cuda": "#FF0000", "cuda_dp": "#FFC000"}
APIS   = ["openmp", "cuda", "cuda_dp"]
LABELS = {"openmp": "OpenMP", "cuda": "CUDA", "cuda_dp": "CUDA DP"}

ORDENS = {
    "mergesort": ["100", "10000", "100000"],
    "quicksort": ["100", "10000", "100000"],
    "bfs":       ["10000x30000", "100000x300000", "500000x1000000"],
    "sssp":      ["1000x4000", "10000x30000", "100000x300000", "200000x400000"],
}
XLABEL = {
    "mergesort": "Tamanho do vetor",
    "quicksort": "Tamanho do vetor",
    "bfs":       "NodosXArestas",
    "sssp":      "NodosXArestas",
}

LARGURA = 0.25

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, sep="|", decimal=",")
df["tamanho"] = df["tamanho"].astype(str)

medias = (
    df.groupby(["algoritmo", "api", "tamanho"])["tempo_total_s"]
    .mean()
    .reset_index()
)

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

for algo, ordem in ORDENS.items():
    sub = medias[medias["algoritmo"] == algo].copy()
    apis_presentes = set(sub["api"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(bottom=0.18)

    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    x = np.arange(len(ordem))
    patches = []

    for i, api in enumerate(APIS):
        valores = []
        for tam in ordem:
            linha = sub[(sub["api"] == api) & (sub["tamanho"] == tam)]
            valores.append(float(linha["tempo_total_s"].values[0]) if len(linha) > 0 else 0.0)
        offset = (i - 1) * LARGURA
        ax.bar(x + offset, valores, LARGURA, color=CORES[api], edgecolor="none", zorder=3)
        if api in apis_presentes:
            patches.append(mpatches.Patch(color=CORES[api], label=LABELS[api]))

    ax.set_xticks(x)
    ax.set_xticklabels(ordem, fontsize=9)
    ax.set_xlabel(XLABEL[algo], fontsize=11)
    ax.set_ylabel("Tempo de execução (segundos)", fontsize=11)
    ax.set_title(f"{algo.capitalize()} — RTX 4090", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(handles=patches, loc="upper right", frameon=False, fontsize=9)

    fig.text(0.02, 0.02, "* CUDA DP: resultados pendentes",
             fontsize=8, color="#888888", ha="left")

    out_path = os.path.join(OUT_DIR, f"tempo_{algo}_rtx4090.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] {out_path}")

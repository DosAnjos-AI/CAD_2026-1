# CLAUDE.md — Regras e Padrões do Projeto CAD_2026-1

Este arquivo define os padrões obrigatórios para todas as sessões deste projeto.
Leia antes de qualquer implementação.

---

## 1. Modo de Teste (Validação Pós-Implementação)

Após cada script implementado, executar **sempre** no modo de teste antes de qualquer outra coisa:

| Parâmetro | Vetores (Mergesort, Quicksort) | Grafos (BFS, SSSP) |
|---|---|---|
| Tamanho | N=100 | 10000 nós × 30000 arestas |
| Execuções | 2 execuções simples | 2 execuções simples |
| Iterações | 1 | 1 |
| Warmup | desativado | desativado |
| Loop completo | NÃO | NÃO |

**Modo de teste não é o benchmark completo.** São apenas 2 execuções diretas para validar compilação e corretude. Nunca rodar o loop completo no modo de teste.

---

## 2. Execução Oficial (Benchmark Completo)

| Parâmetro | Valor |
|---|---|
| Iterações por configuração | 10 |
| Execuções por iteração | 100 |
| Total de execuções | 1.000 |
| Warmup | 1 iteração descartada |
| Sleep entre tamanhos | 30s |
| Sleep entre versões | 20s |

---

## 3. Tamanhos de Entrada — Apenas os do Artigo

**Vetores (Mergesort, Quicksort):**
```
{100, 10000, 100000}
```

**Grafos (BFS):**
```
{10000×30000, 100000×300000, 500000×1000000}
```

**Grafos (SSSP):**
```
{1000×4000, 10000×30000, 100000×300000, 200000×400000}
```

---

## 4. Padrões de Código

- Comentários em português
- Sem emojis em código (prints, logs, comentários)
- Sem código morto — nenhuma função, variável ou include sem uso
- Antes de finalizar qualquer script: verificar código morto explicitamente
- Sem timeout em nenhuma hipótese — nenhum `alarm()`, `signal(SIGALRM)`, ou equivalente

---

## 5. Compilação

| API | Flags obrigatórias |
|---|---|
| OpenMP | `gcc -O2 -fopenmp` |
| CUDA | `nvcc -O2 -arch=sm_61 -ccbin g++-12` |
| CUDA DP | `nvcc -O2 -arch=sm_61 -ccbin g++-12 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -lcudadevrt` |
| Sequencial | `gcc -O2` |

---

## 6. Coleta de Energia

### CPU (ordem de prioridade):
1. `perf stat -e power/energy-pkg/` — primário
   - Testar: `perf stat -e power/energy-pkg/ -- sleep 0 2>/dev/null`
   - Se exit code 0 → disponível
2. `/sys/class/powercap/intel-rapl:0/energy_uj` — fallback
   - Testar: `fopen(...)` — se NULL → indisponível
3. `NA` — se ambos indisponíveis

### GPU:
- RTX 4090 e Jetson: `nvidia-smi dmon -s p -d 100`
- MX350: registrar `NA` diretamente (hardware sem sensor INA)

### Jetson AGX Orin:
- CPU + GPU: `tegrastats --interval 100`

Emitir `[WARN]` no log **uma única vez por sessão** indicando qual método foi selecionado.

---

## 7. Identificação Automática de Hardware

```c
// nvidia-smi --query-gpu=name --format=csv,noheader
// strstr(nome, "MX350")  → "mx350"
// strstr(nome, "4090")   → "rtx4090"
// /proc/device-tree/model contém "Jetson" → "jetson"
// fallback → "desconhecido"
```

---

## 8. Formato do CSV

```
algoritmo|api|versao|hardware|tamanho|iteracao|tempo_total_s|energia_gpu_j|energia_cpu_j|corretude
```

- Separador: `|`
- Decimal: `,` (vírgula)
- Append — nunca sobrescrever linhas existentes
- Skip se `algoritmo|api|versao|hardware|tamanho|iteracao` já existir
- Valores de `versao`: `artigo`, `otimizado`, `sequencial`
- Energia indisponível: `NA`
- Corretude: `OK` ou `FAIL`

---

## 9. Formato do Log

```
[INFO]  YYYY-MM-DD HH:MM:SS mensagem
[WARN]  YYYY-MM-DD HH:MM:SS mensagem
[ERROR] YYYY-MM-DD HH:MM:SS mensagem
```

- Arquivo: `results/benchmark.log`
- Append — nunca sobrescrever
- `[ERROR]` em caso de corretude `FAIL`
- `[WARN]` para energia indisponível (apenas 1x por sessão)

---

## 10. Verificação de Corretude

- Executar em `rep == 0` de **cada** iteração
- Registrar `OK` ou `FAIL` no CSV
- Emitir `[ERROR]` no log se `FAIL`

---

## 11. Estrutura de Pastas

```
CAD_2026-1/
├── CLAUDE.md               <- este arquivo
├── mergesort/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── quicksort/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── bfs/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── sssp/
│   ├── artigo/
│   ├── otimizado/
│   └── sequencial/
├── results/
│   ├── benchmark.csv
│   └── benchmark.log
└── run_all.sh
```

---

## 12. Relatório Final de Cada Sessão

Ao final de toda sessão, trazer obrigatoriamente:
- Status de compilação de cada script
- Resultado dos testes de validação (modo de teste)
- Desvios do plano e justificativas
- Lista de arquivos criados/modificados com caminhos completos
- Confirmação de ausência de código morto
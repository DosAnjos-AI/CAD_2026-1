# HOWTO — Como Executar o Benchmark

---

## Pre-requisitos

Consulte o `REQUIREMENTS.md` para lista completa de dependencias e versoes requeridas.

---

## Permissao RAPL (leitura de energia CPU)

Verificar antes de executar:

```bash
cat /sys/class/powercap/intel-rapl:0/energy_uj
```

Se retornar `Permission denied`:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
```

Esta permissao e perdida apos reboot. Para tornar permanente, veja `REQUIREMENTS.md`.

---

## Clonar e Configurar

```bash
git clone <repo>
cd CAD_2026-1
chmod +x run_all.sh
```

---

## Modo de Teste (validacao rapida)

Editar `run_all.sh` e setar `MODO_TESTE=1` no bloco de configuracao:

```bash
# em run_all.sh, linha MODO_TESTE:
MODO_TESTE=1
```

Executar:

```bash
./run_all.sh
```

Modo teste: 2 execucoes, menor tamanho, sem loop completo. Serve para validar compilacao e corretude antes do benchmark real.

---

## Execucao Completa do Benchmark

Setar `MODO_TESTE=0` em `run_all.sh`, depois:

```bash
# execucao em foreground (terminal deve permanecer aberto)
./run_all.sh

# execucao recomendada — background com nohup (terminal pode ser fechado)
nohup ./run_all.sh >> results/run_all.log 2>&1 &
echo $! > run_all.pid
echo "PID: $(cat run_all.pid)"
```

---

## Monitorar Progresso

```bash
# acompanhar log em tempo real
tail -f results/run_all.log

# verificar se ainda esta rodando
ps -p $(cat run_all.pid) && echo "rodando" || echo "finalizado"

# ver ultimas linhas do CSV
tail results/benchmark.csv
```

---

## Interromper Execucao

```bash
kill $(cat run_all.pid)
```

O script registra a interrupcao no log e para graciosamente.

---

## Retomar Apos Interrupcao

O CSV tem skip automatico — combinacoes ja registradas sao puladas. Basta rodar novamente:

```bash
nohup ./run_all.sh >> results/run_all.log 2>&1 &
echo $! > run_all.pid
```

---

## Verificar Resultados

```bash
# contar linhas coletadas (excluindo header)
tail -n +2 results/benchmark.csv | wc -l

# ver por algoritmo
grep "^mergesort" results/benchmark.csv | wc -l

# verificar falhas de corretude
grep "FAIL" results/benchmark.csv
```

# REQUIREMENTS — Dependencias e Configuracao de Ambiente

---

## Compiladores

- `gcc` >= 10 com suporte a OpenMP (`-fopenmp`)
- `g++` >= 10 (usado como backend pelo `nvcc -ccbin`) — testado com g++-12
- `nvcc` — CUDA Toolkit >= 11.0 recomendado (testado com 12.2)

---

## CUDA

- Driver NVIDIA compativel com a GPU alvo
- Suporte a Dynamic Parallelism (Compute Capability >= 3.5)
- `nvidia-smi` disponivel no PATH

---

## Dependencias por Maquina

| Ferramenta | MX350 | RTX 4090 | Jetson |
|---|---|---|---|
| gcc / g++ | sim | sim | sim (nativo ARM64) |
| nvcc | sim | sim | sim (via JetPack) |
| nvidia-smi | sim | sim | sim |
| tegrastats | nao | nao | sim |
| RAPL sysfs | sim (requer permissao) | sim (requer permissao) | nao |

---

## Permissao RAPL (energia CPU)

Necessaria em cada reboot para leitura de energia via sysfs:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
```

Para tornar permanente via udev:

```bash
echo 'SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj"' \
    | sudo tee /etc/udev/rules.d/99-rapl.rules
```

---

## Espaco em Disco Estimado

- Binarios compilados: ~50 MB
- CSV com dados completos (3 maquinas x 28 scripts x tamanhos x 25 iteracoes): ~5 MB
- Logs: ~10 MB

---

## Sistema Operacional

- Ubuntu 18.04+ ou equivalente Linux x86_64 — para MX350 e RTX 4090
- JetPack 5.x+ — para Jetson AGX Orin (ARM64)

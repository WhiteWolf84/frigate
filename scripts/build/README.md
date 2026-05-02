# Build onnxruntime-vitisai per Phoenix NPU (Linux)

## Panoramica

`onnxruntime-vitisai` non è disponibile su PyPI per Linux. Deve essere compilato
dalla catena sorgente:

```
XRT shim (amd/xdna-driver)
    └─► XIR (graph IR)
         └─► VART (runner abstraction)
              └─► VAIP (VitisAI Inference Processor)
                   └─► onnxruntime + VitisAI EP → wheel Python
```

## Requisiti macchina di build

- Docker installato
- x86_64 (il 5800X3D va benissimo)
- Spazio disco: ~8GB per le sorgenti + build artifacts

## Tempo stimato (5800X3D, j=16)

| Fase | Tempo |
|---|---|
| XRT shim | ~10 min |
| XIR + VART + VAIP | ~20 min |
| onnxruntime wheel | ~60 min |
| Firmware download | ~2 min |
| **Totale** | **~1h 30m** |

## Utilizzo

### Step 1 — Build su macchina con 5800X3D

```bash
# dalla root del repo, sulla tua macchina con 5800X3D:
docker build \
    -f scripts/build/Dockerfile.build \
    -t frigate-npu-builder \
    --build-arg JOBS=$(nproc) \
    .

# Esegui il build e salva gli artifact in scripts/build/output/
mkdir -p scripts/build/output
docker run --rm \
    -v $PWD/scripts/build/output:/output \
    frigate-npu-builder
```

Alla fine trovi in `scripts/build/output/`:
- `onnxruntime_vitisai-*.whl` — wheel Python da installare nel LXC
- `lib/` — librerie `.so` VAIP da copiare nel LXC
- `firmware/1x4.xclbin` — bitstream NPU Phoenix
- `firmware/vaip_config.json` — configurazione VAIP

### Step 2 — Deploy nel LXC Frigate (da host Proxmox)

```bash
# sulla macchina Proxmox (dove gira il LXC):
LXC_ID=<numero_tuo_lxc> bash scripts/build/deploy_to_lxc.sh
```

Lo script copia tutto nel LXC, installa il wheel, configura `LD_LIBRARY_PATH`
e verifica che `VitisAIExecutionProvider` sia visibile.

### Step 3 — Configura Frigate

Aggiungi a `/config/config.yml` nel LXC:

```yaml
detectors:
  npu0:
    type: amd_npu
    xclbin: /opt/xilinx/xrt/amdxdna/1x4.xclbin
    config_file: /opt/vaip/etc/vaip_config.json
    target: AMD_AIE2_Nx4_Overlay
```

```bash
pct exec <LXC_ID> -- systemctl restart frigate
pct exec <LXC_ID> -- journalctl -u frigate -f | grep -i "amd npu\|vitisai"
```

Log atteso se tutto funziona:
```
AMD NPU: loading /config/model_cache/model.onnx [xclbin=..., target=AMD_AIE2_Nx4_Overlay]
AMD NPU: ORT providers actually engaged: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

## Note importanti

- La build avviene in Debian 12 → glibc 2.36 → compatibile con il LXC Frigate
- onnxruntime v1.20.1 corrisponde alla versione trovata nei .so di RyzenAI-SW v1.7.1
- Il firmware `1x4.xclbin` è per Phoenix (AIE2, ryzen12). Non usare i file
  `AMD_AIE2P_*.xclbin` di v1.7.1 che sono per Hawk Point (AIE2P, ryzen14)
- Se il build fallisce su `cmake ../src/shim` in Phase 1, aprire un issue —
  AMD cambia la struttura del repo periodicamente

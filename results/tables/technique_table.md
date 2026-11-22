| Technique | What it does | p50 (ms) | Peak memory (GB) | vs. original |
|---|---|---:|---:|---:|
| native | As shipped (bf16) | 350 | 1.28 | 1.00× |
| fp32_cast | Twice the precision | 331 | 2.50 | 1.06× |
| fp16_autocast | Math in 16-bit | 397 | 1.29 | 0.88× |
| bf16_autocast | Math in bf16 | 360 | 1.28 | 0.97× |
| int8_weight | Weights in 8-bit integers | 468 | 0.70 | 0.75× |

<!-- smolvla, NVIDIA L40S, OCI host, driver 610.57.04, run 64d17ce0. Warm-up 20, 200 timed actions, GPU synchronized. -->

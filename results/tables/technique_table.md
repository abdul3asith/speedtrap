| Technique | What it does | p50 (ms) | Peak memory (GB) | vs. original |
|---|---|---:|---:|---:|
| native | As shipped (bf16) | 622 | 1.28 | 1.00× |
| fp32_cast | Twice the precision | 586 | 2.50 | 1.06× |
| fp16_autocast | Math in 16-bit | 718 | 1.29 | 0.87× |
| bf16_autocast | Math in bf16 | 666 | 1.28 | 0.93× |
| int8_weight | Weights in 8-bit integers | 853 | 0.70 | 0.73× |

<!-- smolvla, NVIDIA L40S, run first-run. Warm-up 20, 200 timed actions, GPU synchronized. Transcribed from terminal log. -->

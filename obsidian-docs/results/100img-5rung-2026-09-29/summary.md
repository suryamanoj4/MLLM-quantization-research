# Ablation Summary

## fp16
- POPE-random: F1 = 0.899, acc = 0.897, yes-ratio = 0.523
- POPE-popular: F1 = 0.851, acc = 0.838, yes-ratio = 0.582
- POPE-adversarial: F1 = 0.807, acc = 0.780, yes-ratio = 0.640
- CHAIR: s = 0.390, i = 0.135
- Attention: mean mass = 0.0997, entropy = 5.493

## w8a8
- POPE-random: F1 = 0.905, acc = 0.905, yes-ratio = 0.505
- POPE-popular: F1 = 0.853, acc = 0.843, yes-ratio = 0.567
- POPE-adversarial: F1 = 0.809, acc = 0.785, yes-ratio = 0.625
- CHAIR: s = 0.460, i = 0.153
- Attention: mean mass = 0.1037, entropy = 5.152

## w4a16
- POPE-random: F1 = 0.888, acc = 0.883, yes-ratio = 0.540
- POPE-popular: F1 = 0.843, acc = 0.828, yes-ratio = 0.595
- POPE-adversarial: F1 = 0.801, acc = 0.770, yes-ratio = 0.653
- CHAIR: s = 0.320, i = 0.115
- Attention: mean mass = 0.0815, entropy = 6.287

## w4a8
- POPE-random: F1 = 0.898, acc = 0.895, yes-ratio = 0.532
- POPE-popular: F1 = 0.851, acc = 0.838, yes-ratio = 0.588
- POPE-adversarial: F1 = 0.801, acc = 0.770, yes-ratio = 0.653
- CHAIR: s = 0.390, i = 0.145
- Attention: mean mass = 0.0823, entropy = 5.998

## w4a4
- POPE-random: F1 = 0.069, acc = 0.508, yes-ratio = 0.028
- POPE-popular: F1 = 0.069, acc = 0.502, yes-ratio = 0.035
- POPE-adversarial: F1 = 0.062, acc = 0.500, yes-ratio = 0.033
- CHAIR: s = 0.030, i = 1.000
- Attention: mean mass = nan, entropy = nan

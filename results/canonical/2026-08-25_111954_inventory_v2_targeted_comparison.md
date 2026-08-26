# Inventory v2 Targeted Comparison

- Domains: `databases, data_engineering`
- Matching: curated_alias + edge_calibrated

## Aggregate (both domains)

| System | Topic F1 | Edge P | Edge R | Edge F1 | Miss Edge | Invalid Extra | Dir Err | Trans Redund | Latency ms | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.598 | 0.216 | 0.338 | 0.262 | 0.650 | 0.710 | 0.024 | 0.008 | 4112 | 0.000288 |
| Prior v1 | 0.731 | 0.072 | 0.107 | 0.083 | 0.757 | 0.853 | 0.136 | 0.010 | 6369 | 0.000469 |
| Prior v2 | 0.759 | 0.317 | 0.486 | 0.367 | 0.479 | 0.588 | 0.071 | 0.082 | 5466 | 0.000447 |

## Edge-level impact (v1 → v2)

- Required recovered: **24**
- Required lost: **4**
- Invalid extras removed: **76**
- Invalid extras added: **32**
- Transitive redundancy Δ: **0.072**

## Per-domain decisions

### databases: **KEEP_V2**

| System | Topic F1 | Edge R | Edge F1 | Invalid Extra | Halluc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.564 | 0.258 | 0.186 | 0.753 | 0.325 |
| Prior v1 | 0.823 | 0.125 | 0.083 | 0.860 | 0.283 |
| Prior v2 | 0.739 | 0.350 | 0.296 | 0.740 | 0.255 |

### data_engineering: **KEEP_V2**

| System | Topic F1 | Edge R | Edge F1 | Invalid Extra | Halluc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.642 | 0.444 | 0.363 | 0.652 | 0.296 |
| Prior v1 | 0.609 | 0.083 | 0.083 | 0.845 | 0.486 |
| Prior v2 | 0.786 | 0.667 | 0.461 | 0.385 | 0.130 |

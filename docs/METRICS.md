# Metric definitions

Let `s` denote a target source cluster, `r` a requested direction in
`{optical, SAR}`, and `k` one of three independently selected opposite
distractors. `y_sr` is the target answer, `d_srk` is the opposite distractor
label, `a_sr0` is the A0 answer, and `a_srk` is the conflict answer.

## Eligibility

All directed metrics condition on an A0-correct source/direction pair:

```text
E_sr = 1[a_sr0 = y_sr]
```

An A0-incorrect pair remains in overall/A0 accuracy and abstention results but
does not enter the CIFR or RSR denominator.

## Pooled CIFR

Replicates are first averaged inside a source/direction:

```text
c_sr = (1/K) * sum_k 1[a_srk = d_srk]
CIFR = mean_{s,r : E_sr=1}(c_sr)
```

Thus three distractors do not create three independent samples.

## RSR

```text
q_sr = (1/K) * sum_k 1[a_srk = y_sr]
RSR = mean_{s,r : E_sr=1}(q_sr)
```

Answers that are neither the target nor distractor label, including abstention,
need not make CIFR plus RSR equal one.

## Directional asymmetry

ASYM is computed only for sources eligible in both directions:

```text
ASYM = mean_s(c_s,SAR - c_s,optical)
```

Positive ASYM means greater observed conflict-induced flipping when SAR is
requested under the fixed protocol. It is a directional interference statistic,
not a causal dominance parameter.

## Robust flip

For three opposite replicates, a source/direction is a robust flip when at least
two answers equal their distractor label. This analysis is secondary.

## Source-cluster bootstrap

Each bootstrap draw samples target sources with replacement. All requested
directions and all conflict replicates for a sampled source travel together.
The reported intervals use 10,000 draws and the pre-specified seed stored in the
dataset protocol configuration.

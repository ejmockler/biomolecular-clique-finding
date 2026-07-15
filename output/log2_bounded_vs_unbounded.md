# Bounded h=2 vs unbounded BFS — fixed-term consistency comparison (log2(x+1))

Intensity scale: `log2(x+1)`  (bounded h=2 = production; unbounded = depth sensitivity)

Scope: `robust`

## Bonferroni-8 pass counts (production log2 scale)

Counts are `bonferroni_pass.sum()` over the 8 discovery-derived terms fixed
before the measured-only/log2 method-transfer reruns
(NPC: nuclear pore, nucleocytoplasmic transport, Vpr-mediated nuclear import;
Splicing: mRNA Splicing, Processing of Capped Intron-Containing Pre-mRNA,
mRNA splicing via spliceosome; Chromatin: chromosome, chromatin), held fixed
across bounded and unbounded. The eightfold raw-p threshold is 0.00625. Because
the terms were selected on this cohort, this is not a post-selection FWER or
selective-inference guarantee.

| contrast | bounded-log2 (h≤2, production) | unbounded-log2 (depth sensitivity) |
|:---------|:------------------------------:|:----------------------------------:|
| c9spor (C9 vs Sporadic, primary) | **8/8** | 6/8 |
| c9ctrl (C9 vs Control)           | **6/8** | 0/8 |
| spctrl (Sporadic vs Control)     | **0/8** | 0/8 |

Bounded-log2 pattern **8/6/0**; unbounded-log2 pattern **6/0/0** (c9spor/c9ctrl/spctrl).

## c9spor — bounded 8/8 vs unbounded 6/8

| cluster   | term                                            |   NES_h2 |   fdr_q_full_h2 | bonferroni_pass_h2   |   NES_unb |   fdr_q_full_unb | bonferroni_pass_unb   |
|:----------|:------------------------------------------------|---------:|----------------:|:---------------------|----------:|-----------------:|:----------------------|
| NPC       | nuclear pore                                    |   1.7902 |          0.0210 | True                 |    1.6019 |           0.0812 | False                 |
| NPC       | nucleocytoplasmic transport                     |   2.2211 |          0.0010 | True                 |    2.4823 |           0.0010 | True                  |
| NPC       | Vpr-mediated nuclear import of PICs             |   1.7119 |          0.0425 | True                 |    1.1234 |           0.5970 | False                 |
| Splicing  | mRNA Splicing                                   |   2.5040 |          0.0010 | True                 |    2.3878 |           0.0010 | True                  |
| Splicing  | Processing of Capped Intron-Containing Pre-mRNA |   2.5881 |          0.0010 | True                 |    2.5705 |           0.0010 | True                  |
| Splicing  | mRNA splicing, via spliceosome                  |   2.4074 |          0.0010 | True                 |    2.3893 |           0.0010 | True                  |
| Chromatin | chromosome                                      |   2.4732 |          0.0010 | True                 |    2.6401 |           0.0010 | True                  |
| Chromatin | chromatin                                       |   2.3448 |          0.0010 | True                 |    2.5899 |           0.0010 | True                  |

## c9ctrl — bounded 6/8 vs unbounded 0/8

| cluster   | term                                            |   NES_h2 |   fdr_q_full_h2 | bonferroni_pass_h2   |   NES_unb |   fdr_q_full_unb | bonferroni_pass_unb   |
|:----------|:------------------------------------------------|---------:|----------------:|:---------------------|----------:|-----------------:|:----------------------|
| NPC       | nuclear pore                                    |   1.8771 |          0.0259 | False                |   -0.7344 |           0.9954 | False                 |
| NPC       | nucleocytoplasmic transport                     |   2.1255 |          0.0031 | True                 |   -0.7061 |           0.9972 | False                 |
| NPC       | Vpr-mediated nuclear import of PICs             |   1.4017 |          0.2970 | False                |   -1.3629 |           0.6202 | False                 |
| Splicing  | mRNA Splicing                                   |   2.4608 |          0.0010 | True                 |    0.4517 |           0.9998 | False                 |
| Splicing  | Processing of Capped Intron-Containing Pre-mRNA |   2.5268 |          0.0010 | True                 |    1.0000 |           0.7993 | False                 |
| Splicing  | mRNA splicing, via spliceosome                  |   2.4372 |          0.0010 | True                 |    0.5150 |           0.9996 | False                 |
| Chromatin | chromosome                                      |   3.0607 |          0.0010 | True                 |    1.0000 |           0.9240 | False                 |
| Chromatin | chromatin                                       |   2.8255 |          0.0010 | True                 |    1.0000 |           0.9240 | False                 |

## spctrl — bounded 0/8 vs unbounded 0/8

| cluster   | term                                            |   NES_h2 |   fdr_q_full_h2 | bonferroni_pass_h2   |   NES_unb |   fdr_q_full_unb | bonferroni_pass_unb   |
|:----------|:------------------------------------------------|---------:|----------------:|:---------------------|----------:|-----------------:|:----------------------|
| NPC       | nuclear pore                                    |   1.6282 |          0.5495 | False                |   -0.8006 |           1.0000 | False                 |
| NPC       | nucleocytoplasmic transport                     |   1.0463 |          1.0000 | False                |   -1.3187 |           1.0000 | False                 |
| NPC       | Vpr-mediated nuclear import of PICs             |   1.4331 |          0.4841 | False                |   -0.7857 |           1.0000 | False                 |
| Splicing  | mRNA Splicing                                   |   1.0169 |          0.9945 | False                |   -1.1980 |           0.9985 | False                 |
| Splicing  | Processing of Capped Intron-Containing Pre-mRNA |   1.1116 |          1.0000 | False                |   -1.2269 |           1.0000 | False                 |
| Splicing  | mRNA splicing, via spliceosome                  |   0.9407 |          1.0000 | False                |   -1.3165 |           1.0000 | False                 |
| Chromatin | chromosome                                      |   0.8373 |          1.0000 | False                |   -0.9908 |           1.0000 | False                 |
| Chromatin | chromatin                                       |   0.9035 |          1.0000 | False                |   -0.7591 |           1.0000 | False                 |

## Interpretation boundary

The primary bounded pattern is **8/6/0** and the unbounded sensitivity is
**6/0/0**. The fixed-term pattern therefore attenuates when the local depth bound
is removed. This supports describing the current result as depth-sensitive and
local to the chosen `h<=2` statistic. It does not prove that bounded depth is
universally correct, identify a deep-shell biological mechanism, establish C9
mutation causality, or turn the same-cohort term panel into independent
confirmation.

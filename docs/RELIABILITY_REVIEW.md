# Reliability review

The existing 87-test CPU suite passed in this audit. A regression test also checks
non-default temporal width through memory, transformer, reference selection and
conditioning; all four must share the configured memory width.

Multi-seed summaries currently aggregate best validation scores. They are not
held-out test results or paired significance tests. Use the full-test evaluation
workflow for publication claims. Confidence intervals now use Student's t
distribution for the actual seed count; one-seed and duplicate-seed experiments
are rejected.

GPU training, real dataset pairing, notebook Run All on RunPod, and multi-seed
research results have not been established by these local CPU checks. Existing
checkpoints must retain their original architecture and configuration.

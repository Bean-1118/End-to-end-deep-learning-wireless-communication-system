# Task 4.5 — Cross-Pair Statistical Summary

Number of Source-to-Target pairs: **3**

## Main findings

- **Full fine-tune** achieved the lowest mean Target BER across the selected PDP pairs (0.167956).
- **Decoder Conv1** was the highest-ranked partial adaptation strategy, with a mean partial-strategy rank of 1.00.
- **Decoder Conv1** provided the strongest mean Target-BER improvement per 100,000 trainable parameters among the partial strategies (0.043100).
- Full fine-tuning should be reported as the best Target-performance strategy, while restricted decoder adaptation should be reported as the preferred parameter-efficient strategy.
- Source forgetting must be discussed alongside Target recovery; a method with stronger Target recovery is not automatically the best deployment choice.

## Reporting caution

The three Source-to-Target pairs are representative experimental conditions, not independent samples from a large statistical population. Therefore, mean, standard deviation and average rank should be described as **cross-pair descriptive statistics**, not as proof of universal statistical significance.

## Compact result table

| Strategy | Mean Target BER | Mean oracle recovery | Mean source forgetting | Mean Target rank | Role |
|---|---:|---:|---:|---:|---|
| No adaptation | 0.253467 | 0.0000 | 0.000000 | 5.00 | Deployment baseline |
| Dense head | 0.245754 | 0.0448 | 0.027466 | 4.33 |  |
| Channel extractor | 0.236560 | 0.0946 | 0.036525 | 3.00 |  |
| Decoder Conv1 | 0.209222 | 0.2376 | 0.034577 | 2.00 | Best partial adaptation; Best partial parameter efficiency |
| Full fine-tune | 0.167956 | 0.4312 | 0.083270 | 1.00 | Best overall Target BER |
| Scratch | 0.301913 | -0.5404 | -0.057981 | 5.67 | Limited-data training-from-scratch control |

## Recommended thesis conclusion

Across the selected cross-PDP deployment scenarios, full-model fine-tuning consistently delivered the strongest Target-domain BER recovery. However, adapting only the first decoder convolution layer was the most effective restricted fine-tuning strategy, providing a substantially better balance between Target recovery, trainable parameter count and Source-domain retention than adapting only the channel-estimation head or the full channel extractor.

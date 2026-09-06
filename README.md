# End-to-end-deep-learning-wireless-communication-system
A multi-task learning framework for sample-efficient adaptation to new environments

This repository contains the experimental code and generated results for an end-to-end learned wireless communication system studied under power delay profile (PDP) mismatch. The work progresses from cross-PDP robustness analysis to transfer learning, parameter localisation, shared-particular optimisation, and few-shot adaptation to unseen channel environments.

The main source code is provided as five archived experiment stages: task2-3.zip, task4.zip, task5.zip, task6.zip, and task7.zip. Other files in the repository are primarily generated experiment outputs, numerical summaries, checkpoints, and figures used for analysis and reporting.

Research workflow

## The experiments follow a staged pipeline:

Task 2-3 — Baseline model, PDP mismatch, and architecture audit
Builds the end-to-end communication model, defines the PDP environments, trains matched-PDP models, evaluates cross-PDP BER, and audits the model architecture and parameter distribution. This stage establishes the mismatch problem: a receiver trained under one PDP can suffer a substantial BER increase when evaluated under a different PDP.

Task 4 — Cross-PDP transfer learning
Tests adaptation strategies with different trainable subsets, including small heads, decoder components, and full-model fine-tuning. Multi-pair experiments compare adaptation behaviour across different source-target PDP combinations.

Task 5 — Parameter localisation
Identifies which decoder parameters are most useful for cross-PDP adaptation. Decoder scans, fine-grained parameter scans, group interaction experiments, and parameter-change audits are used to reduce the adaptable subset while retaining most of the adaptation benefit.

Task 6 — Shared-particular optimisation with iADMM
Splits model parameters into shared and PDP-specific components and applies the implemented iADMM training procedure across historical PDPs. This stage includes parameter-partition checks, rho/sigma searches, multi-PDP training, unseen-PDP adaptation, and strategy comparisons. PDP0-PDP5 are used as historical/training environments and PDP6-PDP7 as unseen environments.

Task 7 — Joint-training baselines and few-shot adaptation
Compares the shared-particular approach against joint-training and adaptation baselines. It evaluates unseen-PDP BER over an SNR grid, studies alternative adaptable decoder subsets, and measures the effect of support-set size on few-shot adaptation.

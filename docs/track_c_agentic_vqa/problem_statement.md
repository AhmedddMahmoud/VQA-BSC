# Problem Statement — Track C Agentic VQA

## Motivation

Track A established a classical baseline and Track B established a strong pretrained VLM baseline. The next research question is whether a **carefully scoped agentic wrapper** can recover part of the remaining errors without destabilizing the current system.

## Why Track B Is Already Strong

Track B delivers high performance across scaling points:

- 1800: exact/top1 `0.8556`, soft `0.9287`
- 5000: exact/top1 `0.8474`, soft `0.9242`
- 10000: exact/top1 `0.8446`, soft `0.9225`

Therefore, Track B remains the primary modern branch and the reference model for future extensions.

## Why Agentic Behavior Could Still Help

Even strong VLMs can fail on subsets that require precise counting, relation disambiguation, localized detail extraction, or text-sensitive interpretation. A lightweight planner/router may selectively invoke better prompts or re-check logic for these cases.

## What “Agentic VQA” Means in This Project

In this repository, Agentic VQA means:

- using the existing Track B answerer as the base engine,
- adding a small decision layer that routes questions to specialized prompt strategies,
- optionally running a second-pass verification,
- returning a final answer with a route trace.

## What Problem Track C Is Trying to Solve

Track C aims to test whether a compact routing-and-recheck wrapper can improve accuracy on **remaining Track B failure modes**, especially:

- counting / quantity questions,
- ambiguity and annotation-sensitive questions,
- object localization/detail-sensitive questions,
- potentially OCR-sensitive questions,
- spatial / relation-sensitive questions.

This is explicitly an **exploratory, prototype-level** objective.

## What Problem Track C Is NOT Trying to Solve

- It is **not** replacing PaliGemma or Track B.
- It is **not** a full RL-based multi-tool learning program.
- It is **not** immediate full-dataset re-architecture.
- It is **not** proof that agentic methods will always improve performance.

Track C is a controlled extension study, not a reset of the project direction.

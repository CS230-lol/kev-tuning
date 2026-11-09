---
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: adapter
library_name: peft
tags: [kev, decision-model, typesafe, system-one, calibration]
---

# CS230-lol/kev-0.8b-jigsaw-toxicity

A [Kev](https://github.com/jaredpalmer/kev) decision model fine-tuned from `jaredpalmer/kev-0.8b` on 2547 labelled records
for one workload, with a temperature (1.04) fitted on 449 held-out records so the served probabilities
are calibrated by default.

| development (7000 questions) | jaredpalmer/kev-0.8b (calibrated) | this model (calibrated) |
| --- | --- | --- |
| accuracy | 0.586 | 0.814 |
| brier | 0.527 | 0.267 |
| ece | 0.050 | 0.010 |
| coverage at 5% error | 0.00 | 0.49 |

Serve it with the Kev repo (`uv run --extra serve python -m kev.serve --run CS230-lol/kev-0.8b-jigsaw-toxicity`) or the `kev-finetune` skill's Modal
endpoint. Trained with `kev.train --init_from` (LoRA delta + pointer head) at kev commit `fc4a17e194bc`; `result.json`,
`training_config.json` and `train.log` are in this repo.

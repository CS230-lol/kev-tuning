---
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
library_name: peft
tags: [kev, decision-model, typesafe, system-one, calibration]
---

# CS230-lol/kev-4b-jigsaw-toxicity

A [Kev](https://github.com/jaredpalmer/kev) decision model fine-tuned from `jaredpalmer/kev-4b` on 2547 labelled records
for one workload, with a temperature (1.11) fitted on 449 held-out records so the served probabilities
are calibrated by default.

| development (7000 questions) | jaredpalmer/kev-4b (calibrated) | this model (calibrated) |
| --- | --- | --- |
| accuracy | 0.759 | 0.838 |
| brier | 0.335 | 0.231 |
| ece | 0.029 | 0.008 |
| coverage at 5% error | 0.28 | 0.62 |

Serve it with the Kev repo (`uv run --extra serve python -m kev.serve --run CS230-lol/kev-4b-jigsaw-toxicity`) or the `kev-finetune` skill's Modal
endpoint. Trained with `kev.train --init_from` (LoRA delta + pointer head) at kev commit `fc4a17e194bc`; `result.json`,
`training_config.json` and `train.log` are in this repo.

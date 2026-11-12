# Language-model agents' beliefs can be manipulated via truthful evidence in open channel

This repository contains the code and datasets for our study of belief manipulation in language-model agents through truthful evidence in open channels such as social media. Coordinated agents arrange truthful fragments into misleading narratives, prompting victim agents to accept conclusions that the evidence does not establish.

The study examines the threat across models and events, traces its effects on downstream decisions, and tests which fragments influence agents' judgements. These mechanism analyses motivate Grounded Monitor (GM), a proof-of-concept neuro-symbolic runtime monitor that checks evidential support to help agents reconsider unsupported conclusions.

## Modules

| Module | Research focus |
| --- | --- |
| [Lying with Truths](lying_with_truth/README.md) | Generative Montage, victim-agent evaluation, downstream decisions and CoPHEME |
| [Mechanism analysis](mechanism_analysis/README.md) | Fragment responses, deletion and transplantation experiments, and synthetic narrative controls |
| [Monitoring](monitoring/README.md) | Grounded Monitor support assessment and decision-accuracy evaluation |

## Getting started

```bash
git clone https://github.com/CharlesJW222/Narrative-Overfitting-NMI.git
cd Narrative-Overfitting-NMI
```

Each module has its own README with setup and usage instructions. Follow the instructions in the relevant directory to run its experiments.

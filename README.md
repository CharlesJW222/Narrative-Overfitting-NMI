# Narrative Overfitting

Research repository for **Language-model agents can be manipulated via truthful evidence in open channel**, prepared for submission to *Nature Machine Intelligence*.

The project studies how language-model agents combine truthful fragments into misleading narratives, which fragments shape their judgements, and how explicit evidence-support monitoring can help them reconsider unsupported conclusions.

## Repository structure

```text
Narrative-Overfitting-NMI/
├── lying_with_truth/       Original Generative Montage code, data and README
├── mechanism_analysis/     Fragment interventions and synthetic narrative controls
├── monitoring/             Grounded Continuation and monitoring evaluation
├── CONTRIBUTING.md         Module integration guidance
└── README.md               Project overview
```

| Module | Contents | Status |
| --- | --- | --- |
| [Lying with Truths](lying_with_truth/README.md) | Original generation code, victim and downstream evaluation, CoPHEME files and example attack plans | Imported from the original repository |
| [Mechanism analysis](mechanism_analysis/README.md) | Fragment responses, payload interventions and controlled narrative comparisons | Awaiting contributor upload |
| [Monitoring](monitoring/README.md) | GC support assessment and monitoring evaluation | Awaiting contributor upload |

The manuscript, cover letter and manuscript figures remain in Overleaf. This repository currently contains the original release and integration directories for the additional research modules.

## Original project

The full contents of [CharlesJW222/Lying_with_Truth](https://github.com/CharlesJW222/Lying_with_Truth) have been moved into `lying_with_truth/`, including its original README. The import preserves all 35 commits through [`21183d3`](https://github.com/CharlesJW222/Lying_with_Truth/commit/21183d3ce578d7d4bc05c51e29857e9e97237bfa); the original repository is unchanged.

To work with the inherited entrypoints, start from their own directory:

```bash
git clone https://github.com/CharlesJW222/Narrative-Overfitting-NMI.git
cd Narrative-Overfitting-NMI/lying_with_truth
```

See [the original README](lying_with_truth/README.md) for that release's documentation and citation. Its scripts and environment snapshots are preserved as published; this import does not establish complete reproduction of the NMI experiments.

The original work appeared as an ACL 2026 Outstanding Paper:

- [Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage](https://aclanthology.org/2026.acl-long.270/).

## Adding the NMI modules

Contributors can upload the mechanism-analysis and monitoring implementations into their respective directories. Each module should provide its own dependencies, configurations and instructions for reproducing the relevant experiments. See [CONTRIBUTING.md](CONTRIBUTING.md) for the directory conventions.

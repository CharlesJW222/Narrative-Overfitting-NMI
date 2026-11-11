# Contributing research modules

The repository separates the original release from the NMI mechanism-analysis and monitoring implementations.

| Directory | Responsibility |
| --- | --- |
| `lying_with_truth/` | Original Generative Montage release and its documentation |
| `mechanism_analysis/` | Fragment readouts, interventions and synthetic narrative controls |
| `monitoring/` | GC implementation, response evaluation and associated controls |

Upload each implementation into its designated directory. Keep its dependency specification, configuration files and README alongside the code, and document the working directory expected by each command. Reference shared assets in `lying_with_truth/` explicitly instead of silently copying or changing them.

A module README should explain the required environment, model or API configuration, input data, commands for each experiment, expected output files and correspondence with manuscript figures or tables. Record seeds, model versions and metric definitions where they affect reproducibility. State which commands have actually been tested.

Use environment variables or local ignored configuration for credentials. Keep generated runs and model caches out of commits unless they are deliberately selected research assets.

Submit changes on a branch and open a pull request describing the implementation and validation. Update the root README's integration status when a module is added. The Overleaf manuscript and submission correspondence remain separate from this public repository.

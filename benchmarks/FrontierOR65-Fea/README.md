# FrontierOR65-Fea

This self-contained suite contains 65 feasible FrontierOR task contracts selected by
problem class and formulation type. The repository stores each task's problem statement,
metadata, schema, checker, and evaluation support files. Large instances and reference
solutions remain outside Git and are supplied through `--large-root`.

Source: [SmartOR/FrontierOR](https://huggingface.co/datasets/SmartOR/FrontierOR),
licensed under CC BY 4.0. DecisionBrain selected and reorganized these task contracts;
see the repository's `THIRD_PARTY_NOTICES.md` for attribution and modification details.

Expected external layout:

```text
<large_root>/<paper_id>/instance/large_instance_<n>.json
<large_root>/<paper_id>/gurobi_solution/large_solution_<n>.json
```

Run the complete suite from the repository root:

```powershell
python -m decisionbrain.benchmark.runner --suite FrontierOR65-Fea --large-root <path>
```

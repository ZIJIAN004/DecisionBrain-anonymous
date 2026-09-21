# FrontierOR10-Inf

This self-contained suite contains ten infeasible FrontierOR variants derived from five task
families. For each family, one variant changes instance data and one changes a hard constraint.
Public problem statements do not disclose the expected infeasibility result.

Source: [SmartOR/FrontierOR](https://huggingface.co/datasets/SmartOR/FrontierOR), licensed under
CC BY 4.0. DecisionBrain created these evaluation variants and records their provenance in
`index.json`. See the repository's `THIRD_PARTY_NOTICES.md` for attribution and modification
details.

External instances use the case IDs in `index.json`:

```text
<large_root>/<case_id>/instance/large_instance_<n>.json
```

Run the suite from the repository root:

```bash
python -m decisionbrain.benchmark.runner --suite FrontierOR10-Inf --large-root /path/to/data
```

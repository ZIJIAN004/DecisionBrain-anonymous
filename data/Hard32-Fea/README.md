# Hard32-Fea local data

This directory is the canonical local data root for the `Hard32-Fea` benchmark.
The repository does not redistribute the complete third-party instance values.

The tracked files in this directory are limited to these instructions and
`sources.lock.json` plus `prepare.py`. Downloaded and reconstructed data is
ignored by Git.

## Prepare the instances

From the repository root, run:

```bash
python data/Hard32-Fea/prepare.py
```

The script downloads only the selected source instances from the commits pinned
in `sources.lock.json`, converts them to the DecisionBrain task schemas, applies
the fixed deadlines and fleet caps from `selection.json`, and verifies the
expected byte size of all 32 generated JSON files. Use `--force` to replace an
existing generated dataset.

This command does not repeat candidate screening, solve the instances, or
generate reference solutions. Those are experimental artifacts; the final
selection and task parameters are already fixed in `selection.json`.

## Expected layout

```text
data/Hard32-Fea/
  sources.lock.json
  downloads/                       # pinned upstream files or repositories
  work/                            # temporary conversion files
  instances/
    jssp_deadline/instance/large_instance_<n>.json
    vrptw_minfleet/instance/large_instance_<n>.json
    pdptw_minfleet/instance/large_instance_<n>.json
  solutions/
    jssp_deadline/gurobi_solution/large_solution_<n>.json
    vrptw_minfleet/gurobi_solution/large_solution_<n>.json
    pdptw_minfleet/gurobi_solution/large_solution_<n>.json
```

The benchmark runner's `--large-root` argument should point to
`data/Hard32-Fea/instances`. Reference solutions are kept separately because
they are evaluation artifacts rather than participant inputs.

The 32-case order and DecisionBrain-specific limits are defined in
`benchmarks/Hard32-Fea/selection.json`. The task descriptions, schemas, and
checkers are defined in the corresponding directories under
`benchmarks/Hard32-Fea/`.

`sources.lock.json` records provenance; it is not a claim that the original
benchmark authors granted redistribution rights. Users must comply with the
upstream terms when downloading the source data.

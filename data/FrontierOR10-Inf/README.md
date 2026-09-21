# FrontierOR10-Inf data

This directory contains the ten derived infeasible instances used by the
`FrontierOR10-Inf` suite. They are tracked directly because the complete payload
is small enough for ordinary Git and the source FrontierOR dataset is licensed
under CC BY 4.0.

Each case directory follows the runner's external-data layout:

```text
data/FrontierOR10-Inf/<case_id>/instance/large_instance_<n>.json
```

Run the suite with:

```bash
python -m decisionbrain.benchmark.runner \
  --suite FrontierOR10-Inf \
  --large-root data/FrontierOR10-Inf
```

Case provenance and whether the variant changes data or a hard constraint are
recorded in `benchmarks/FrontierOR10-Inf/index.json`.

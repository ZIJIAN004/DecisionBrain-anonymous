# FrontierOR65-Fea data

The 65-case selection and task contracts are tracked under
`benchmarks/FrontierOR65-Fea/`. The instance payload is sourced from the
SmartOR/FrontierOR dataset, licensed under CC BY 4.0.

The payload is not stored in ordinary Git because the suite index includes very
large JSON files. GitHub rejects individual files larger than 100 MiB, and
versioning generated copies would unnecessarily inflate every clone. This is a
transport limitation, not a redistribution-rights restriction.

Expected local layout:

```text
data/FrontierOR65-Fea/dataset/
  <paper_id>/instance/large_instance_<n>.json
  <paper_id>/gurobi_solution/large_solution_<n>.json
```

Verify all 65 upstream instance and reference-solution records without
downloading their contents:

```powershell
python data/FrontierOR65-Fea/fetch.py
```

This checks the pinned revision, remote paths, and instance byte sizes, and
writes `remote_manifest.json` containing upstream Git/LFS/Xet identifiers. The
manifest can be committed after a successful complete verification. To fetch
the payload only when an evaluation machine needs it:

```powershell
python data/FrontierOR65-Fea/fetch.py --download
```

Run the suite with:

```powershell
python -m decisionbrain.benchmark.runner `
  --suite FrontierOR65-Fea `
  --large-root data/FrontierOR65-Fea/dataset
```

The pinned source revision is the final 2026-06-14 snapshot before the upstream
September republication and is recorded in `sources.lock.json`. It retains the
`<paper_id>/instance/` layout used to construct this suite. Do not substitute
the incomplete local `FrontierOR_large` directory for the published 65-case
dataset: every file must be checked against `benchmarks/FrontierOR65-Fea/index.json`.

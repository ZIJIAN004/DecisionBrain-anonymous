# Benchmark data

This directory is the local root for benchmark data that is not distributed in
the Git repository. The benchmark contracts, checkers, schemas, and fixed case
selection remain under `benchmarks/`.

Dataset publication policy:

- [`FrontierOR65-Fea`](FrontierOR65-Fea/README.md): tracked manifest and
  acquisition metadata; large CC BY 4.0 payload fetched locally.
- [`FrontierOR10-Inf`](FrontierOR10-Inf/README.md): all ten derived instance
  files are tracked directly.
- [`Hard32-Fea`](Hard32-Fea/README.md): tracked manifest and reconstruction
  metadata; third-party source values reconstructed locally because the
  redistribution chain is not sufficiently explicit.

Runtime outputs and reference solutions are not source data and should not be
committed. Each dataset directory explains which inputs are tracked and why.

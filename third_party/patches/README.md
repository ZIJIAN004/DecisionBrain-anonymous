# Third-Party Patches

This directory contains source patches required to reproduce the baseline
integrations used by DecisionBrain. Each patch applies to a fixed upstream
revision and remains subject to the upstream project's license.

## OptiMUS v2 FrontierOR Integration

- Upstream repository: <https://github.com/teshnizi/OptiMUS>
- Upstream branch: `optimus-v0.2`
- Base revision: `181cd82b55f2d7396940511bbc46c6bdb17aab1a`
- Patch: `optimus-v2-frontieror.patch`
- Patch SHA-256: `390973ce29a4744de99d0f297a1ec725c6673dc9dd8b0ec203f5ca7e0c089321`
- License: MIT License; retain the upstream copyright and license notice.

The patch adds the FrontierOR input adapter, frozen conversion validation,
parallel scheduling, Linux bubblewrap/systemd resource isolation, incumbent
capture, fixed result formatting, and OpenAI-compatible model configuration.
It also exposes the Gurobi solve limit through `ADAPTER_SOLVER_TIMEOUT`, with a
30-second default for the reported FrontierOR evaluation. The complete Agent
task deadline defaults to 7200 seconds through `ADAPTER_TASK_TIMEOUT`.

Apply the patch from a clean OptiMUS checkout:

```bash
git clone https://github.com/teshnizi/OptiMUS.git OptiMUS-v2
cd OptiMUS-v2
git checkout 181cd82b55f2d7396940511bbc46c6bdb17aab1a
git apply --check ../DecisionBrain/third_party/patches/optimus-v2-frontieror.patch
git apply ../DecisionBrain/third_party/patches/optimus-v2-frontieror.patch
```

Before running the adapter, configure the dataset and runtime paths explicitly
when they differ from the portable defaults under `external/`:

```bash
export FRONTIEROR_INDEX=/path/to/FrontierOR65-Fea/index.json
export FRONTIEROR_INSTANCE_ROOT=/path/to/FrontierOR65-Fea/dataset
export FRONTIEROR_PROBLEM_ROOT=/path/to/FrontierOR65-Fea/contracts
export ADAPTER_PYTHON_ENV=/path/to/python/environment
export ADAPTER_CGROUP_SLICE=optimus-evaluation.slice
export ADAPTER_SOLVER_TIMEOUT=30
export ADAPTER_TASK_TIMEOUT=7200
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

`GUROBI_HOME` is optional when the Gurobi installation does not require a
separate read-only bind inside the sandbox. API credentials must be supplied
through environment variables and must not be committed.

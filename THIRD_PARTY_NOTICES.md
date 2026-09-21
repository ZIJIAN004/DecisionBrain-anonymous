# Third-Party Notices

DecisionBrain is licensed under the Apache License 2.0, except for the third-party
materials identified below. Each third-party work remains subject to its own copyright
and license terms. The DecisionBrain license does not grant additional rights to those
materials.

## Materials Distributed in This Repository

### FrontierOR

- Project: FrontierOR
- Source: <https://huggingface.co/datasets/SmartOR/FrontierOR>
- Paper: <https://arxiv.org/abs/2605.25246>
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Location: `benchmarks/FrontierOR65-Fea` and `benchmarks/FrontierOR10-Inf`
- Changes: DecisionBrain selects and reorganizes task metadata and contracts for its
  evaluation suites, adds runner-specific metadata, and keeps large instances and reference
  solutions outside this repository. Some suites contain evaluation-only schemas, checkers,
  formulations, and reference code derived from the upstream dataset.

Users who redistribute or adapt these materials must comply with CC BY 4.0, including its
attribution and indication-of-changes requirements. Individual benchmark tasks may also be
based on the publications named by their task identifiers and metadata; users should cite the
relevant original publications when using those tasks.

### Hard32-Fea source instances

The `benchmarks/Hard32-Fea` contracts select or transform instances from three established
benchmark families. Large instances and reference solutions are not distributed in this
repository.

- `jssp_deadline`: Taillard job-shop instances, with a synthesized common deadline not present
  in the original benchmark. Source publication: <https://doi.org/10.1016/0377-2217(93)90182-M>.
- `vrptw_minfleet`: Gehring and Homberger VRPTW instances, with a fleet cap derived from the
  SINTEF best-known solution tables. Source: <https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/>.
- `pdptw_minfleet`: Li and Lim PDPTW instances, with a synthesized fleet cap. Source:
  <https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/>.

The DecisionBrain task descriptions, selection records, and checkers document the changes made
for Hard32-Fea. Users obtaining the original instances must comply with the terms published by
their respective providers.

## External Experimental Baselines

The following baseline repositories were used in comparative experiments. Their complete source
trees are not redistributed here. A DecisionBrain integration patch is provided only for OptiMUS
v2, whose upstream MIT License permits redistribution. Listing a project below is an attribution
and reproducibility record, not a grant of rights beyond the applicable upstream license.

### COOPA

- Project: COOPA: A Modular LLM Agent Architecture for Operations Research Problems
- Upstream source: <https://github.com/xxxxxa-hub/COOPA>
- Upstream revision used as the adaptation base: `e96ddd5689502dda74e531fcd0df1e86e8ca253e`
- License status: No license file or explicit software license was identified in the upstream
  repository at the recorded revision.
- Local changes: FrontierOR adapters, sandboxing, scheduling, result formatting, execution
  limits, and related integration changes were developed separately for the experiments.

No right to copy, modify, or redistribute COOPA is granted by DecisionBrain. Obtain any
required permission from the upstream copyright holders before redistributing its source or a
derived patch.

### OptiMUS v2

- Project: OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models
- Upstream source: <https://github.com/teshnizi/OptiMUS/tree/optimus-v0.2>
- Upstream revision used as the adaptation base: `181cd82b55f2d7396940511bbc46c6bdb17aab1a`
- License: MIT License, copyright (c) 2023 Ali Teshnizi
- Paper: <https://arxiv.org/abs/2402.10172>
- Local changes: FrontierOR adapters, sandboxing, scheduling, validation, result formatting,
  execution limits, and related integration changes were developed separately for the
  experiments.
- Distributed patch: `third_party/patches/optimus-v2-frontieror.patch`, applied to the upstream
  revision identified above. Application and configuration instructions are in
  `third_party/patches/README.md`.

OptiMUS v2 may be modified and redistributed under its MIT License provided that its copyright
and permission notice are retained. DecisionBrain distributes the integration patch, not a
complete copy of the OptiMUS source tree.

### OR-LLM-Agent

- Project: OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization
  Problems with Reasoning Large Language Models
- Upstream source: <https://github.com/bwz96sco/or_llm_agent>
- Upstream revision used as the adaptation base: `4d5306fe20a83d6eca9ea857e0afa1137f592987`
- License status: No license file or explicit software license was identified in the upstream
  repository at the recorded revision.
- Paper: <https://arxiv.org/abs/2503.10009>
- Local changes: FrontierOR adapters, sandboxing, scheduling, checker integration, result
  adaptation, execution limits, and related integration changes were developed separately for
  the experiments.

No right to copy, modify, or redistribute OR-LLM-Agent is granted by DecisionBrain. Obtain any
required permission from the upstream copyright holders before redistributing its source or a
derived patch.

## Runtime Dependencies

DecisionBrain also depends on third-party Python packages and optional optimization solvers.
Those packages are not vendored in this repository and retain their respective licenses. See
`pyproject.toml` for the dependency list. Commercial or restricted components, including
Gurobi, require users to obtain and comply with their own licenses.

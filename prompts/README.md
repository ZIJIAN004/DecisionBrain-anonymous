# 提示词目录（prompts/）

优化建模求解 Agent 的提示词分两层维护：

```
prompts/
├── stage_common_system.txt 通用阶段提示词层：workspace 纪律与非终止进度工具
├── algorithm_library_system.txt 算法库提示词层：仅注入 algorithm_design 与 solving
├── intake_system.txt       intake 阶段系统提示词层
├── problem_contract_system.txt 问题契约系统提示词层
├── algorithm_design_system.txt 系统提示词层
├── solving_system.txt        系统提示词层
├── feasibility_review_system.txt 独立可行性验收与责任判断
├── explain_system.txt      系统提示词层
└── runtime/                运行契约层：输出格式 / 时间预算 / 解析约定
    ├── intake_contract.txt
    ├── problem_contract_contract.txt
    ├── algorithm_design_contract.txt
    ├── solving_contract.txt
    ├── feasibility_review_contract.txt
    └── explain_contract.txt
```

`src/decisionbrain/core/prompts.py` 启动时加载当前 StageAgent 流水线需要的提示词。`debug_*`、`acceptance_*` 等历史提示词文件可以保留在目录中作为素材，但不参与当前 Core 装配。改完任何运行中使用的文件需重启服务生效。

## 当前流水线

澄清问题定义 → 问题契约 → 问题契约审计 → 算法设计 → 工具驱动求解 → 独立可行性验收 → 结果解释。

问题契约审计是算法设计前的二值门禁。审计员通过 workspace 工具完整读取
`problem.md`、确认后的问题定义、真实数据、两个 schema 和 `feasibility_checker.py`：
`approve` 才进入算法设计，`revise` 会携带结构化 findings 退回问题契约阶段重建全部契约文件。

可行性验收阶段通过无参数固定工具执行 checker，再独立决定是否放行；接受后直接进入结果解释，Solving 不读取 checker 或结果。拒绝时只判断责任属于实例、算法设计或求解实现。Core 将 `algorithm_design` 和 `solving` 责任确定性路由到对应阶段。原始 evidence 保存在 Core 私有状态中；不含 checker 细节的语义修复指令随阶段提示完整下发，执行 workspace 只生成 `feasibility_handoffs/<target>/attempt{N}/`，向责任阶段提供旧 solution 等大体积基线，实际写出的基线随本轮是否产生候选而变，且只保留当前轮——早期轮次的求解产物由 Solving 发布的不可变 artifact 承担。最新指令完整下发，早期历史仅保留稳定 ID 账本；Feasibility Reviewer 可按 attempt 取回单轮私有审计。`instance` 暂时保持终止行为。

阶段之间通过 workspace 固定文件流转：`stage_outputs/*.json`、`input_schema.json`、`solution_schema.json`、`feasibility_checker.py`、`input.json`、`solution.json`、`feasibility_result.json` 和 `solver_result.json`。

## 打磨须知

- 系统提示词只写 agent 的职责、专长和工作原则，不写 JSON 字段名、后端解析、时间预算、RESULT_JSON 等运行细节。
- 运行契约才写输出格式、字段名、解析约定和时间预算。
- 澄清专家只围绕目标、约束、决策澄清业务语义；数据追问只服务于这三者。
- `solving_contract.txt` 的 `__TIME_BUDGET__` 由 `decisionbrain/core/agent.py` 按后端超时自动注入，不要在系统提示词里写死时间。

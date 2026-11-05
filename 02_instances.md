# 02 — 实例构造

状态：**已实现（2026-10-03）**。生成器：[toy/build_two_stage_toy.py](toy/build_two_stage_toy.py)；
实例：`toy/two_stage_toy.json`（1.3 MB，生成物，权威实例）。

---

## 1. 场景 = 全天轨迹

| | model_v2 / EXP-BATTLVL | 两阶段 |
|---|---|---|
| 随机对象 | stage 1 固定为 $\mathrm{round}(\lambda^1)$；之后每阶段 $N_b$ 个原子，重组 | $K_s$ 条**完整轨迹** $\xi^s=(d^{1,s},\ldots,d^{T,s})$，**$d^1$ 也随机** |
| 概率 | 每原子 $1/N_b$ | 每条 $1/K_s$ |
| stagewise independence | 必需 | 不需要（以后可以放跨期相关，例如日总量共享） |

确定性部分（$n=3,T=4$、12 辆车、6 名工人、$R,C_p,d,c,\phi$）和 Poisson 率 $\lambda^t_{ij}$ 都从 `reference/model_v2_toy/toy_instance.json`
（model_v2 toy 的冻结拷贝）**加载时读取**，不在生成器里重抄。所以两阶段 toy 与 v2 toy 只差随机结构。

## 2. 抽样规则

沿用 model_v2 的抽样器（直接 import `build_toy_tree.poisson_icdf`）：

- 逆 CDF，每次抽取恰好消耗一个均匀数（$\lambda=0$ 也消耗）；
- **每条轨迹一条随机子流**，种子 `pool_seed·100000 + k`；轨迹内按 $t$、再 $(i,j)$ 升序；
- 于是任意前 $K_s$ 条就是一个合法的 $K_s$ 场景样本，且不同 $K_s$ 互为前缀。

两个独立的池：

| 池 | 种子 | 大小 | 用途 |
|---|---|---:|---|
| `train` | 20261003 | 1000 | SAA 问题（取前 $K_s$ 条；或切成不相交的块做 $M$ 组复制） |
| `eval` | 20261004 | 5000 | 固定价格的 out-of-sample 精确评估 |

生成结果（核验通过）：

| 变体 | 池 | (SC-5) 最大节点需求 | 每天平均出行 |
|---|---|---:|---:|
| toy-A（$\lambda_{ii}=0$） | train / eval | 10 / 10 ≤ 12 | 16.31 / 16.37 |
| toy-B（$\lambda_{ii}=0.15$） | train / eval | 10 / 10 ≤ 12 | 18.18 / 18.16 |

## 3. 价格与常数

JSON 的 `pricing`：菜单 $\{0,\ldots,8\}$、基准 4、三种结构（`uniform` / `type` / `od`，见 [01 §1](01_model.md)）。结构在求解时选择。

`constants`：$Q_1=6$、$Q_3=12$、$Q_2=\bar P+\bar d+\bar c=12$。

电池参数在**加载时**计算（[src/two_stage_bl.jl](src/two_stage_bl.jl) 的 `load_ts_instance`，复用 `reference/battery_levels.jl` 的
`level_transition` / `round_to_grid`）：$\theta_L=0.2$，$\Delta_{ij}=0.8\,\phi_{ij}=0.12$，概率网格 1/20。

| 档数 | 掉档概率 | Γ |
|---|---|---:|
| K=2（= model_v2） | 0.15 | 20 |
| K=3（H/M/L，默认） | 0.3 | 10 |

`d_bar` 与 `max_node_demand` 也写进 JSON，但模型本身已经不需要它们（没有状态界、没有割）；只用于核验与报告。

## 4. 规模

K=3、`type` 结构，Gurobi 报告的 100 场景 EF：

| 对象 | 规模 |
|---|---|
| 一个场景（4 期） | 约 1 260 变量（约 250 二元）、1 620 行 |
| EF，100 场景 | 126 018 变量（25 218 二元）、162 002 行；presolve 后 59 593 变量 |
| EF，1000 场景 | 约 126 万变量 —— 预计需要 decomposition，见 [03](03_algorithms.md) |

## 5. 怎么扩展

- **换 $K_s$**：`load_ts_instance(...; nscen = Ks)` 取 train 池前 $K_s$ 条，不需要重新生成。
- **SAA 复制**：把 train 池切成 $M$ 个不相交块（例如 10 × 100）。
- **更大网络**：生成器只依赖确定性块与 $\lambda$，换输入即可；规模会被第二阶段的稳定匹配块（$n^3$ 个 $\delta,\eta$ 每期）主导。

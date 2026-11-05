# model_2stage — 定价 + 全天需求已知的两阶段 LIME-OPR

**已实现并通过冒烟测试（2026-10-03）。** 2026-10-03 从 `lime-opr/model_2stage/` 整体迁入 `sddip/model_2stage/`，`lime-opr` 已废弃；本目录不再依赖它，用到的旧文件都冻结拷贝在 [reference/](reference/)。

| 阶段 | 内容 |
|---|---|
| 第一阶段 | **任务定价** $p_{jk}$（离散菜单），在看到任何需求之前决定，全天不变 |
| 第二阶段 | 全天需求 $(d^1,\ldots,d^T)$ **同时已知**；每个场景一个 T 期确定性 MILP |
| 物理模型 | 电池分档（[reference/battlvl_FORMULATION.md](reference/battlvl_FORMULATION.md)）：K 档、H-first、每档取整、调度只搬满电车、换电 K→1 |
| 算法 | 不用 SDDiP；枚举 / EF / （待写）integer L-shaped |

SDDiP 需要的设定全部删掉：二值化状态、local copy、A2 弹性松弛、initial cut、重组场景树、确定性根节点、SDDP.jl。
Γ 只为 floor 规则保留。

---

## 文件

| 文件 | 内容 |
|---|---|
| [01_model.md](01_model.md) | 数学模型：价格如何进入工资与稳定匹配、精确线性化、信息结构、删除与保留清单 |
| [02_instances.md](02_instances.md) | 实例构造：全天轨迹、抽样规则、两个场景池、价格菜单、规模 |
| [03_algorithms.md](03_algorithms.md) | 求解算法：方法对比、推荐路线、测试结果、风险 |
| [04_algorithm.md](04_algorithm.md) | **baseline 算法**：价格空间分支定界 + 固定价格精确求值（pre / lazy）+ 无稳定匹配的松弛上界 |
| [05_model_v3.md](05_model_v3.md) | **新模型 v3 约束草稿**（2026-10-17 起，未实现）：5 档整数掉档；搬车/换电两组工人、pin 软容量、区域定价待写。2026-10-30 起不再继续 |
| [int_flow/](int_flow/README.md) | **当前模型（2026-10-31 起，TODO #1）**：没有电池分级、骑行车流全为整数（每个需求按固定种子抽"骑完会不会没电"），每天各自的初始状态，φ 按换电次数标定；n = 30 实例；测试 [test/test_int_flow.jl](test/test_int_flow.jl)。固定价格下的分解（TODO #3，2026-11-02：对 x 列生成 + fix 步、Benders）见 [int_flow/README.md](int_flow/README.md#todo-3分解2026-11-02-起) |
| [scaling/](scaling/README.md) | **规模实验**：n、T、场景数对求解性能的影响；合成实例生成器、运行脚本、剪枝、结论 |
| [realdata/](../weekly/2026-10-10_10-16/文档/model_2stage/realdata/README.md)（2026-10-17 归档，脚本已删除） | **真实数据实验**（2026-10-13）：巴黎 kmeans_w+repair 实例，n 10–50 × 时长 2–18 h × K_s 10–30；稀疏场景模型 [src/sparse_scenario.jl](src/sparse_scenario.jl) |
| [fixed_price/](fixed_price/README.md) | **固定价格下的求解**（2026-10-06）：T 的时间分解（relax-and-fix）、稳定匹配的代价与 R1 + tie-break + 检查 |
| [toy/build_two_stage_toy.py](toy/build_two_stage_toy.py) | 实例生成器 → `toy/two_stage_toy.json` |
| [src/two_stage_bl.jl](src/two_stage_bl.jl) | 模型：`load_ts_instance`、`add_prices!`、`add_scenario!`、`build_ef`、`evaluate_prices`、`conservation_residual` |
| [test/smoke_test.jl](test/smoke_test.jl) | 正确性测试（HiGHS，不需要 Gurobi） |
| [test/test_baseline.jl](test/test_baseline.jl) / [slurm_baseline_test.sh](slurm_baseline_test.sh) | baseline 的正确性测试（需要 Gurobi） |
| [src/price_bnb.jl](src/price_bnb.jl) | 外层价格空间分支定界 |
| [run_grid.jl](run_grid.jl) / [slurm_grid.sh](slurm_grid.sh) | 枚举全部价格（`uniform` / `type`），精确 SAA 最优 + out-of-sample |
| [run_ef.jl](run_ef.jl) / [slurm_ef.sh](slurm_ef.sh) | 两阶段 EF（任意结构）+ out-of-sample |
| [slurm_smoke.sh](slurm_smoke.sh) | 冒烟测试的 sbatch（HiGHS，不占 Gurobi 会话） |
| `results/<job id>/` | 每个作业的结果（JSON / CSV）；本地运行写到 `results/local/` |
| `slurm_logs/` | 作业日志（已过滤许可证信息） |
| `env/` | Julia 环境（`Project.toml` + `Manifest.toml`，拷贝自原 `lime-opr/sddip`，包版本不变） |
| [reference/](reference/) | 从 lime-opr 冻结拷贝的依赖，见下表 |

`reference/` 的内容（2026-10-03 拷贝，除注明处外未改动）：

| 文件 | 来源（lime-opr 内） | 用途 |
|---|---|---|
| `model_v2_toy/toy_instance.json` | `model_v2/toy/` | 确定性数据与 Poisson 率 |
| `model_v2_toy/build_toy_tree.py` | `model_v2/toy/` | 生成器复用其中的 `poisson_icdf` |
| `battery_levels.jl` | `experiments/EXP-BATTLVL-battery-levels/` | 档位转移函数；冒烟测试 T1 的参照实现。**只改了 include 路径** |
| `model_v2.jl` | `sddip/src/v2/` | `battery_levels.jl` 依赖它 |
| `battlvl_FORMULATION.md` | `experiments/EXP-BATTLVL-battery-levels/FORMULATION.md` | 电池分档的约束定义 |
| `model_v2_01_problem_definition.md` | `model_v2/01_problem_definition.md` | 底层业务模型的完整约束 |

## 怎么跑

在 `/global/home/users/zjl0512/sddip` 下运行。求解与测试都走 sbatch（见 `Agent.md` §3.2）。

```bash
python3 model_2stage/toy/build_two_stage_toy.py          # 生成实例（几秒，可在登录节点）
sbatch model_2stage/slurm_smoke.sh                       # 冒烟测试，约 3 分钟
sbatch --time=04:00:00 --export=ALL,RUNS="toy-A:3:type:100" model_2stage/slurm_grid.sh
sbatch --time=06:00:00 --export=ALL,RUNS="toy-A:3:od:100"   model_2stage/slurm_ef.sh
```

RUNS 格式为 `variant:档数:价格结构:场景数`，多个用 `+` 连接。Gurobi WLS 最多 **2 个并发会话**，同时最多开 2 个作业。

---

## 结果

全部为 toy-A、K=3（H/M/L，Γ=10）、菜单 $\{0,\ldots,8\}$，利润口径。

### 正确性（[test/smoke_test.jl](test/smoke_test.jl)，全部通过）

- 第二阶段 MILP 与 EXP-BATTLVL 的 SDDP.jl 模型在同一条全天轨迹上的确定性等价**逐场景相等**（K=2 与 K=3）。
- 每个解出的场景逐期守恒残差为 0。
- `type` 结构 3 场景：EF 最优 = 81 点枚举最优 = 150.333。

### `type` 结构、100 场景，枚举（精确，406 s；job 39595529，`results/39595529/`）

| 调度价 | 1 | **2** | 3 | 4（v2 基准） | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| 样本内平均利润 | 133.55 | **140.74** | 137.16 | 133.58 | 130.00 | 119.27 |

| 价格 | 样本内（100） | 样本外（eval 池 1000 条） |
|---|---:|---:|
| SAA 最优：调度 2 | 140.74 | **137.16 ± 1.70** |
| v2 基准：全部 4 | 133.58 | 129.73 ± 1.66 |

读法：

1. **调度价 2 恰好是距离 1 的工人的接单阈值**（$d+c=1+1$）。价 1 时只有同节点工人接单；价高于 2 不会多接单，只多付工资
   （每涨 1 元利润少约 3.6，即每天约 3.6 个调度任务）。
2. **换电价格在 toy 上完全无关**：每个调度价下，9 个换电价的利润都相同，即任何换电价都没有被执行的换电。
   一天只有 4 期、约 16 次出行，几乎没有车掉到 L 档，而换电要占 3 期。要研究换电定价需要更长的一天或更高的耗电。
3. 相对基准价 4，最优价在样本外多赚约 7.4（5.7%），两个置信区间不重叠。样本内比样本外高约 3.6，是 100 场景 SAA 的乐观偏差。

### EF（`type` 与 `od`，100 场景，Gurobi 4 线程、MIPGap=0、时限 3600 s；job 39595489，`results/39595489/`）

| 结构 | 状态 | incumbent | 上界 | gap | 样本外（1000 条） |
|---|---|---:|---:|---:|---:|
| `type` | TIME_LIMIT | 140.74 | 143.01 | 1.6% | 137.16 ± 1.70 |
| `od` | TIME_LIMIT | 141.17 | 143.69 | 1.8% | 137.50 ± 1.69 |

- `type`：3600 s 都没能证明最优；incumbent 等于枚举的精确最优 140.74（2 分钟时还停在 133.55，约 11 分钟才找到）。
  toy 上低维价格结构**应该用枚举，不用 EF**。
- `od` 的 incumbent 价格：调度 1→2、2→1 为 1，其余调度为 2，换电都是 4（换电价无关）。样本内只比 `type` 高 0.43，
  样本外 137.50 vs 137.16，差距在置信区间之内。toy 上按 OD 分别定价没有带来可分辨的收益。

---

## 下一步

0. **（2026-10-04）baseline 算法与规模实验已完成**，结论与下一步建议见 [scaling/README.md](scaling/README.md#结论)。
   **（2026-10-06）固定价格下的单场景求解**见 [fixed_price/README.md](fixed_price/README.md#结论)：稳定性代价、R1 + tie-break 的精确捷径、relax-and-fix。
1. **`od` 结构**：EF 1 小时关不掉 gap（1.8%）。要证明最优就写 integer L-shaped（[03 §3](03_algorithms.md)）；但 toy 上 `od` 相对 `type` 的收益在样本外不可分辨，优先级不高。
2. **SAA 复制**：train 池切成 10 × 100，逐组枚举，给出统计上界与最优价格的稳定性。
3. **让换电有意义**：更长的 T 或更高的耗电，否则 `type` 结构里的换电价是摆设。
4. **价格下限问题**：外部选项为 0 且 $u=0$ 视为接单，价格会被压到阈值上；是否加保留效用需要你决定（[03 §5](03_algorithms.md)）。
5. **平台直接指派工人（不用 stable matching）**：平台直接决定 $x_{ijk}$，与稳定匹配下的结果对比。
   - 做法：在 `add_scenario!` 加开关 `matching = :stable | :assign | :assign_ir`。`:assign` 删掉 (S1)–(S6)（$v,s,\delta,\eta,\zeta$），
     只保留 (M2)(M3)(W1)：每期少 $2n^3+n$ 个二元变量。
   - 注意：只删 (S1)–(S6) 会连工人的个体理性一起删掉，价格就只剩"付工资"一个作用，最优价必然是菜单最低价 0。所以要同时跑
     `:assign_ir`：平台指派，但工人可以拒绝亏本任务，即 $x_{ijk}>0\Rightarrow p_{jk}\ge d_{ij}+c_{jk}$
     （用 $x_{ijk}\le Q_1\eta_{ijk}$ 加 $p_{jk}-d_{ij}-c_{jk}\ge-Q_2(1-\eta_{ijk})$ 实现）。
   - 要回答的问题：(a) 稳定性的代价 = `:assign_ir` 与 `:stable` 的利润差，以及最优价格变不变；(b) 去掉稳定匹配块后 EF / 枚举快多少；
     (c) 两种指派下的调度量与失需在哪里不同。
   - 先在 toy-A、K=3、`type` 结构、100 场景上用枚举跑，直接和上面的结果表对比。

## 参考文献

- Laporte & Louveaux (1993), *The integer L-shaped method for stochastic integer programs with complete recourse*, Oper. Res. Lett.
- Carøe & Schultz (1999), *Dual decomposition in stochastic integer programming*, Oper. Res. Lett.
- Angulo, Ahmed & Dey (2016), *Improving the integer L-shaped method*, INFORMS J. Comput.
- Zou, Ahmed & Sun (2019), *Stochastic dual dynamic integer programming*, Math. Program.
- Mak, Morton & Wood (1999), *Monte Carlo bounding techniques for determining solution quality in stochastic programs*, Oper. Res. Lett.

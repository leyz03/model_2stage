# int_flow — 没有电池分级、骑行车流全为整数的模型（TODO #1）

2026-10-31 起。决定见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#1 的写法"。
本文件记模型改动、实例和标定。没提到的部分与 [01_model.md](../01_model.md) 相同。

## 模型

- **电量只有两种状态：** 可租（A）和没电（U）。读取时 `levels = 2`，Γ = 1，所以 (D1) 变成 I = A，floor 规则不起作用。
- **需求拆分（数据）：** 场景 s 的需求 d_ij^t 拆成两部分：
  - scenU[s][t][i, j]：骑完车会没电的乘客；
  - 其余是骑完车仍可租的乘客。
- **约束：**
  - YU_ij^t ≤ scenU：服务了的、骑完车会没电的乘客，整数；
  - Yod_ij^t − YU_ij^t ≤ d − scenU：服务了的、骑完车仍可租的乘客；
  - 服务 YU 的车在 t + c_ij 到 j 时进 U，其余的车进 A。
  - 骑行车流按构造就是整数，不再有期望流拆分。
- **初始状态：** 每个场景用当天 08:00 的数据。A0 是所有可租的车，U0 是低电量候选车；08:00 正在骑行中的车不计入。
  Q3（车队规模）也按场景取。
- **建模方式：** 场景之间的初始状态和车队规模不同，所以只在 `scenario_view(inst, s)` 上建模，用它的场景 1。
  直接在多场景实例上调 `add_scenario!` / `add_scenario_sparse!` 会报错，不会悄悄用错初始状态。
  时间分解的 `window_instance(inst, s, ta, tb)` 已经走 `scenario_view`。
- **启发式初始解：** `nothing_start!`、`continue_start!` 缺车时先服务骑完车仍可租的乘客。
- **新的乐观假设：** 缺车时，平台可以选择先服务哪一类乘客，所以会先服务骑完车仍可租的。报结果时要和 01_model §4 的乐观假设一起写明。
- **(inst, s) 形式的入口：** `solve_tier_sparse`、`solve_scenario`、`relax_and_fix` 等要传 `scenario_view(inst, s)` 和 s = 1。
  `window_instance` 和时间分解（`add_td_block!`）可以直接传原实例和 s。带价格的 `build_ef`、`price_bnb` 还没改（#4 时改）。

## 拆分规则：共用随机数，固定种子

- 用种子 20261031 给每个 (t, i, j) 抽一串均匀数 u_1, u_2, …，所有场景共用这一串。
- 第 k 个乘客（k ≤ d_ij^t）骑完车没电，当且仅当 u_k < φ_ij。
- 结果：
  - 需求数相同，拆分就相同；需求更多，没电的乘客不会更少；
  - 随机的只有需求和每天的初始状态。
- 拆分结果写进实例文件，复现时不依赖 numpy 的版本。

## 文件

| 文件 | 内容 |
|---|---|
| [build_int_instance.py](build_int_instance.py) | 从 build_instances.py 的实例生成拆分实例：`scenarios_U`、每天的 `A0` / `U0`、元数据 `int_split` |
| [calibrate_phi_swaps.py](calibrate_phi_swaps.py) | φ 按换电次数标定：所有 φ_ij 乘同一个系数 |
| `calibration/phi_swaps_all.json`、`calibration/phi_swaps_workers.json` | 标定结果（basis = all / workers） |
| `instances/paris_n30_H10_int_raw.json` | 未标定（φ 用 40 英里续航的标定） |
| `instances/paris_n30_H10_int.json` | φ 按全部换电次数标定（α = 0.527） |
| `instances/paris_n30_H10_int_workers.json` | **用这个**（11-01 起）：φ 只按工人换电标定（α = 0.416），每天骑完车没电的乘客 929.3（期望 902.1） |
| [../test/test_int_flow.jl](../test/test_int_flow.jl)、[../slurm_int_flow_test.sh](../slurm_int_flow_test.sh) | 正确性测试 |
| [partial_int.jl](partial_int.jl)、[run_partial_int.jl](run_partial_int.jl)、[slurm_partial_int.sh](slurm_partial_int.sh)、[summarize_partial_int.py](summarize_partial_int.py) | TODO #2：只对部分变量保留整数约束（full / xbin / x / lp），放松后的单纯形重解，汇总表；结果在 `results/<job>/` |
| [../test/test_partial_int.jl](../test/test_partial_int.jl)、[slurm_partial_int_test.sh](slurm_partial_int_test.sh) | #2 的正确性测试（HiGHS） |
| [alt_price.jl](alt_price.jl)、[run_alt_price.jl](run_alt_price.jl)、[slurm_alt_price.sh](slurm_alt_price.sh) | TODO #4：按 OD 定价的交替求解（O 步、P 步、上界），见下面"TODO #4：交替求解"；结果在 `results/<job>/alt_T<T>.json` |
| [../test/test_alt_price.jl](../test/test_alt_price.jl)、[slurm_alt_price_test.sh](slurm_alt_price_test.sh) | #4 的正确性测试（Gurobi） |
| [decomp.jl](decomp.jl)、[run_decomp.jl](run_decomp.jl)、[slurm_decomp.sh](slurm_decomp.sh)、[summarize_decomp.py](summarize_decomp.py) | TODO #3：固定价格下的两种分解，对 x 列生成（`cg_solve`）、Benders（`benders_solve`），见下面"TODO #3：分解"；结果在 `results/<job>/` |
| [../test/test_decomp.jl](../test/test_decomp.jl)、[slurm_decomp_test.sh](slurm_decomp_test.sh) | #3 的正确性测试（Gurobi） |
| [COLUMN_GENERATION.md](COLUMN_GENERATION.md) | 列生成 `cg_solve` 的流程（什么时候解 LP、MIP，什么时候停）、松 / 紧实例上各步的表现、待定的细节和决定（2026-11-03 起） |
| [scale_fleet.py](scale_fleet.py) | TODO #7：车队按 ρ 缩小（每天的 A0、U0，最大余数法取整），其余不变 |
| [gap_at_time.py](gap_at_time.py) | 从 Gurobi 日志读 decomp 各任务在某个时刻的解、界和 gap（相当于更短的时限） |
| `instances/paris_n30_H10_int_workers_fleet050.json`、`..._fleet030.json` | TODO #7 的紧实例：ρ = 0.5、0.3 |
| [lazy_stab.jl](lazy_stab.jl)、[summarize_lazy_stab.py](summarize_lazy_stab.py) | TODO #9：稳定匹配行按需生成（`run_decomp.jl --stab lazy`，full / lp / cg），见下面"TODO #9：稳定匹配按需生成" |
| [../test/test_lazy_stab.jl](../test/test_lazy_stab.jl) | #9 的正确性测试（Gurobi）：`sbatch --export=ALL,TEST=test_lazy_stab.jl model_2stage/int_flow/slurm_decomp_test.sh` |

三个实例都来自 `spatial_clustering/results/39957895/instances/paris_n30_H10.json`
（n = 30，08:00–18:00，6 分钟一期，T = 100，6 月 30 天各一个场景，工人 34 人，R、C_p 是合成值）。

```bash
cd /global/home/users/zjl0512/sddip
B=spatial_clustering/results/39957895
.venv/bin/python -I model_2stage/int_flow/calibrate_phi_swaps.py --instance $B/instances/paris_n30_H10.json \
    --out model_2stage/int_flow/calibration/phi_swaps_all.json
.venv/bin/python -I model_2stage/int_flow/build_int_instance.py --instance $B/instances/paris_n30_H10.json \
    --initial-state $B/n30/initial_state.npz --calibration model_2stage/int_flow/calibration/phi_swaps_all.json \
    --out model_2stage/int_flow/instances/paris_n30_H10_int.json
# 只按工人换电（11-01 起用这个）：上面两步加 --basis workers，输出 phi_swaps_workers.json、paris_n30_H10_int_workers.json
```

Julia 里：`inst = load_ts_instance(path, "paris"; levels = 2)`，然后 `v = scenario_view(inst, s)`，再用
`solve_tier_sparse(v, 1, p, opt)` 等求解。

## 标定（2026-10-31）

- **数据：** `paris_tasks_2025-06.csv`。条件是 BatterySwapTask、电动车、已完成（done）、collection 发生在研究域内（labels.csv 的 H3 r9 格子），时间在 08:00–18:00。

  | | 每天平均 |
  |---|---:|
  | 研究域内的骑行 | 40,739 |
  | 换电，全部 | 1,142.2 |
  | 换电，只算工人（juicer、ops） | 902.1 |
  | 骑完车没电的期望值（φ 按 40 英里续航） | 2,168.4 |

- **系数：**
  - 按全部换电：α = 0.5268，按骑行加权的 φ 从 0.0532 降到 0.0280，即每次骑行约耗 2.2% 电，平均约 36 次骑行没电；
  - 只按工人换电：α = 0.4160。
- **抽样后的实际值：** 每天骑完车没电的乘客，标定后的实例是 1,174.0（期望 1,142.2），未标定的是 2,220.9（期望 2,168.4）。
  共用随机数下，抽样偏差对 30 天是同一个，平均后不会抵消。
- **没有计入：** 低电量车数在 08–18 点之间的变化。

## TODO #3：分解（2026-11-02 起）

决定和结果见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#3 分解"。代码 [decomp.jl](decomp.jl)。
两种方法都精确，都基于 #2 的结论：x 和全部二元变量固定后，剩下的是网络 LP，顶点是整数。

- **列生成（`cg_solve`）：** 受限问题 = 完整的 tier 模型，未激活的 x 上界设为 0（行全保留，是原问题的限制）。
  - 初始列：d_ij = 0 的组合（工人在任务所在节点），价格 4 下每期 620 / 4,310。
  - LP 段：解受限 LP，读未激活 x 的 reduced cost，> 1e-6 的激活，直到没有。最后的 LP 值就是全量 LP 值 UB。
  - 激活列时用 `activate!` 保住单纯形基：Gurobi 把 ub = 0、rc > 0 的列记成"在上界"，只放开上界会让下一次从 x = ub 开始
    （T = 100 加 37 列要 17k 次 dual simplex、45 s）；改成标为"在下界"后用 primal simplex 接着解（0.4 s）。
  - fix 步，三级，每级时限 600 s，第一个达到 UB 的就是最优：
    - `:all`：LP 解里取整数值的 x、β、z 固定，只对小数的部分求 MIP；
    - `:zero`：只固定 LP 里为 0 的 x（在 LP 解的支撑上求 MIP，二元变量都放开）；
    - `:face`：只固定 rc ≠ 0 的 x（LP = MIP 时，值为 UB 的解都在 LP 最优面上，这些 x 取同一个界）。
  - 都没到 UB：解受限 MIP；还到不了 UB 就做 reduced-cost fixing（x_et ≥ 1 的解最多值 UB + rc_et），把 rc ≥ LB − UB 的列加进来再解，精确。
  - 利润是整数，LB > UB − 1 也算证明。
- **Benders（`benders_solve`）：** 主问题变量 v = (x, β, z)，主问题 = 只含 v 和工人流（W、Gin）的行 + η ≤ 割；子问题 = 整个模型的 LP，v 固定。
  - 最优性割：η ≤ Q(v̄) + g·(v − v̄)，g 是固定变量的 reduced cost；
  - 可行性割：子问题不可行时用 Farkas 射线（Gurobi `InfUnbdInfo = 1`），y0 + y·v ≤ 0；
  - 先 LP 段（主问题放松）到 η = Q，再 MIP 段（每轮重解主问题 MIP）。
  - 每轮有 JuMP 开销（逐个读对偶、Farkas 的变量界部分），n = 30 上每轮约 0.8–5 s。T = 10、50 上它比 Gurobi 时间大 6–18 倍；T = 20 上主问题 MIP 变慢，开销约为 Gurobi 时间的一半。所以结果表同时报 Gurobi 时间和墙钟。
  - 11-02 的结果：可行性割占 94–99%；T = 20 上 Gurobi 时间是 full 的 84–89 倍（06-11 到 2 h 没证明），T = 50 到 2 h 还在 LP 段。见周 TODO "#3 分解的结果"。
- **紧实例（TODO #7，车队缩到 ρ = 0.5、0.3）：** LP > MIP，fix 步三级都到不了 LP 界；T = 100 上一次激活 8.7–11.7 万列后，
  `activate!` + primal simplex 的重解比直接解全量 LP 慢 10 倍以上，`cg_solve` 在 LP 段超时报错退出，不写结果。见周 TODO "#7 结果"。

```bash
cd /global/home/users/zjl0512/sddip
sbatch model_2stage/int_flow/slurm_decomp_test.sh                                                  # 测试，约 3 分钟
sbatch --array=0-35 --export=ALL,METHODS="full lp cg",OPTS="--tl 7200" model_2stage/int_flow/slurm_decomp.sh
sbatch --array=0-8 --export=ALL,METHODS="benders",TS="10 20 50",OPTS="--tl 7200" model_2stage/int_flow/slurm_decomp.sh
python3 model_2stage/int_flow/summarize_decomp.py 40748007 40748494 40749460 40748120              # 11-02 的结果表
# TODO #7：车队缩到 ρ = 0.5 / 0.3 的实例上重跑（ρ = 0.3 把 050 换成 030）
.venv/bin/python -I model_2stage/int_flow/scale_fleet.py --instance model_2stage/int_flow/instances/paris_n30_H10_int_workers.json \
    --rho 0.5 --out model_2stage/int_flow/instances/paris_n30_H10_int_workers_fleet050.json
INST=model_2stage/int_flow/instances/paris_n30_H10_int_workers_fleet050.json
sbatch --array=0-35 --export=ALL,METHODS="full lp cg",OPTS="--instance $INST --tl 7200" model_2stage/int_flow/slurm_decomp.sh
sbatch --array=0-11 --export=ALL,METHODS="cg",OPTS="--instance $INST --tl 7200 --fixstep 0" model_2stage/int_flow/slurm_decomp.sh
```

## TODO #9：稳定匹配按需生成（2026-11-03 起）

决定和结果见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#9 稳定匹配按需生成"。代码 [lazy_stab.jl](lazy_stab.jl)。

- **组：** (i, 档 h, t) 一组，即一个 z、一行工人一侧的行、i 在这一档的全部任务池一侧的行，和 `add_tier_direct!` 写给这个 z 的行完全一样。只加一部分组是原模型的松弛。
- **LP 检查（闭式）：** lo = (W − Σx)/Q1，hi = 1 − max_e (M − H_e)/Q3；max(lo, 0) ≤ min(hi, 1) 时有 z ∈ [0, 1] 满足这一组，否则加这一组。
- **整数检查：** 工人一侧的行逼 z = 1（W − Σx ≥ 1），同时任务池一侧的某行逼 z = 0（M − H_e ≥ 1）。和 `blocking_pairs_sparse` 的条件一致（测试 L2）。
- **`--stab lazy` 的三种用法：**
  - `full`：建模时不带稳定匹配行，z 全部建好；lazy 回调在每个整数候选解上加违反的组，一次分支定界，精确。
  - `lp`：行生成。先解不带这些行的 LP，加违反的组，用 dual simplex 重解，直到没有违反。
  - `cg`：LP 段没有 rc > 0 的列时查行，加行后用 dual simplex 重解，直到既没有列也没有行；MIP 段（fix 步、受限 MIP、reduced-cost fixing）用 lazy 回调，回调加过的组在下一次求解前变成普通行。
  - `cg --stab defer`（11-04）：LP 段同上；进 MIP 前把剩下的组一次全部加成普通行，不用回调（结果标签 `_deferM`）。
- **为什么 cg 仍然精确：** LP 段结束时，没加的行对偶取 0，它们的 z 的 reduced cost 也是 0，所以这组对偶对全量 LP 可行，LP 值就是全量 LP 的值，reduced-cost fixing 照样成立。

```bash
cd /global/home/users/zjl0512/sddip
sbatch --export=ALL,TEST=test_lazy_stab.jl model_2stage/int_flow/slurm_decomp_test.sh             # 测试
INST=/global/home/users/zjl0512/sddip/model_2stage/int_flow/instances/paris_n30_H10_int_workers_fleet050.json
sbatch --array=0-26 --export=ALL,METHODS="full lp cg",TS="20 50 100",OPTS="--instance $INST --stab lazy" model_2stage/int_flow/slurm_decomp.sh
sbatch --array=0-8 --export=ALL,METHODS="cg",TS="20 50 100",OPTS="--instance $INST --stab lazy --fixstep 0" model_2stage/int_flow/slurm_decomp.sh
sbatch --array=0-8 --export=ALL,METHODS="cg",TS="20 50 100",OPTS="--instance $INST --stab defer" model_2stage/int_flow/slurm_decomp.sh   # 11-04
python3 model_2stage/int_flow/summarize_lazy_stab.py <原写法的作业> <lazy 的作业> <defer 的作业>     # tier → lazy 对照表；cg 另有 tier / lazy / defer 表
```

## TODO #10：拉格朗日时间分解（2026-11-04 起）

决定和结果见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#10 拉格朗日时间分解"。代码 [lag_td.jl](lag_td.jl)，
沿用 [../src/lagrangian_td.jl](../src/lagrangian_td.jl)（`form = :sparse`）的分块、副本和联合 LP 乘子。

- **上界：** L(λ) = Σ_k max { 块 k 的利润 + λ_k·y_k − λ_{k−1}·z_k } ≥ 最优值，对任意 λ 成立；块限时停下或块放松了整数性时，用它已证明的界，仍然成立。
- **一次评估：** λ 取联合 LP（各块用副本等式连起来）的对偶，此时 LP 放松的块之和 = LP 值；然后每个块单独一个任务求解。
- **块的写法（`--form`）：** `full` 全整数；`xbin` 只保留 x 和 β、z（和全整数同值，#2）；`bin` 只保留 β、z（块的松弛）；`lp` 全部放松（含入状态副本，用来核对 Σ = LP 值）。前三种的入状态副本都是整数。
- 测试 [../test/test_lag_td_int.jl](../test/test_lag_td_int.jl)：145 项全部通过（job 40883634）。

```bash
cd /global/home/users/zjl0512/sddip
sbatch --export=ALL,TEST=test_lag_td_int.jl model_2stage/int_flow/slurm_decomp_test.sh            # 测试
INST=/global/home/users/zjl0512/sddip/model_2stage/int_flow/instances/paris_n30_H10_int_workers_fleet050.json
LPJ=$(sbatch --parsable --export=ALL,PHASE=lp,DAY=11,T=100,B=5,OPTS="--instance $INST" model_2stage/int_flow/slurm_lag_td.sh)
sbatch --array=0-19 --dependency=afterok:$LPJ --export=ALL,PHASE=block,DAY=11,T=100,B=5,FORMS="full xbin bin lp",\
LAM=/global/home/users/zjl0512/sddip/model_2stage/int_flow/results/$LPJ/lambda_d11_T100_B5.json,OPTS="--instance $INST --tl 1800" \
    model_2stage/int_flow/slurm_lag_td.sh
python3 -I model_2stage/int_flow/summarize_lag_td.py <lp 作业> <块作业> --full 40788834/d11_T100_full \
    --sol 40788834/d11_T100_full --sol 40836624/d11_T100_xbin --sol 40856647/d11_T100_cg_deferM --sol 40856648/d11_T100_cgnofix_deferM
```

## TODO #4：交替求解（2026-11-01 起）

决定见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#4 交替求解"。代码 [alt_price.jl](alt_price.jl)。

- **价格：** 每个 OD（含换电 j = k）一个价，从自己的档位里选（`menus[j, k]`）。第一轮实验所有 OD 的档位都是菜单 0–12。
- **O 步：** 固定价格，各场景用 tier 写法精确求解（`ostep`，同 `solve_tier_sparse`）。从上一轮的运营出发（`start_from!`）。
- **P 步：** 固定所有场景、所有期的 x、W、M，求让它们仍是稳定匹配的最低价格（`price_rows` + `least_prices`）。
  - 记 H_iab = Σ_{i2: d_i2a ≤ d_ia} x_i2ab。对每个有工人的节点 i：
    - i 接了 jk：p_jk ≥ d_ij + c_jk；
    - i 接了 jk，任务池 ab 对 i 没满（H_iab < M_ab）：p_jk − p_ab ≥ (d_ij + c_jk) − (d_ia + c_ab)；
    - i 有闲置工人，ab 对 i 没满：p_ab ≤ d_ia + c_ab。
  - 这和 `blocking_pairs_sparse` 的定义加个体理性等价。
  - 都是差分约束，可行集对逐分量取 min 封闭，有最小元。最小元对任何任务量都使工资最低，从下界往上推就能求出，和 m̃ 的数值无关。
  - 最小元 ≤ 当前价格，所以旧运营在新价格下仍可行、工资不增；O 步解到最优时利润单调不减。
  - 没人做的 OD：`:keep` 保持原价；`:follow` 先让它跟着有人做的 OD 一起降，再在约束允许的范围内抬回去，但不超过原价。
- **起点：** 先枚举 uniform 价格，从统一价出发。从 uniform 最优出发时，结果不差于 uniform 最优。
- **上界（`discrim_ub`）：** 去掉稳定匹配，每个工人按自己门槛 d_ij + c_jk 向上取到该 OD 的档位付钱，任务最高允许到最高档。
  - 对档位上的任何 OD 价格，最优运营里接单的工人都满足 p_jk ≥ d_ij + c_jk，所付工资不低于这个档位，所以这个值是上界。
  - 没有价格变量，各场景分开解。
  - 松在两处：同一个 OD 的所有工人、所有期、所有场景必须付同一个价；稳定匹配被去掉。

### 第一轮结果（2026-11-01，job 40722065、40722487、40722503）

3 天（11、18、1 日）平均利润，档位 0–12。表和结论见 [weekly/2026-10-31_11-06/TODO.md](../../weekly/2026-10-31_11-06/TODO.md) "#4 第一轮结果"。
汇总命令：`.venv/bin/python -I model_2stage/int_flow/summarize_alt_price.py 40722065 40722487 40722503`。

- 上界在 T ≤ 20 上等于统一价 1，在 T = 50 上只高 1.00：这些时段上价格基本不起作用。
- T = 100 上交替（从统一价 3 出发）比统一价最优高 13.67，离上界还差 79.33，差的全是工资。
- 计算量：价格 12 允许 25,903 个工人-任务组合（价格 4：4,310）。T = 10 上统一价 12 的 3 天求解共 1,056 s，统一价 4 共 19 s。
  上界每个时段 1.5–10 分钟（T = 100：581 s，10 GB）。

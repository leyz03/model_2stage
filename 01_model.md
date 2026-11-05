# 01 — 数学模型：定价 + 全天需求已知的两阶段模型

状态：**working specification（2026-10-03）**。实现：[src/two_stage_bl.jl](src/two_stage_bl.jl)。

物理底稿是电池分档模型 [reference/battlvl_FORMULATION.md](reference/battlvl_FORMULATION.md)（原 lime-opr EXP-BATTLVL）
（它又以 [reference/model_v2_01_problem_definition.md](reference/model_v2_01_problem_definition.md) 为底稿，下文简称 model_v2）。本文件只写**改了什么**；没提到的约束与两份底稿逐字相同。

---

## 0. 已定的设定（2026-10-03）

| 项 | 设定 |
|---|---|
| 第一阶段 | **任务定价** $p_{jk}$，在看到任何需求**之前**决定，全天不变 |
| 第二阶段 | 全天需求 $\xi=(d^1,\ldots,d^T)$ **同时已知**，做 stage $1..T$ 的全部运营决策 |
| 物理模型 | 电池分档：$K$ 档、H-first 骑行、每档单独取整、调度只搬 1 档车（`move=:top`）、换电 $K\to1$ |
| SDDiP 装置 | **全部删除**（§5） |

与上一版分析稿的区别：$d^1$ 不再是确定性根节点，而是场景的一部分；第一阶段不再是 stage 1 的运营决策，而是价格。

---

## 1. 第一阶段：价格

价格从离散菜单 $\mathcal P=\{P_1,\ldots,P_L\}$ 中选（toy：$\{0,1,\ldots,8\}$，v2 的基准价 4 在菜单里）。
任务 $(j,k)$ 属于价格组 $g(j,k)$，同组同价：

| 结构 | 组 | 第一阶段二元变量数（toy，$L=9$） |
|---|---|---:|
| `uniform` | 全部任务一个价 | 9 |
| `type` | 调度（$j\ne k$）一个价、换电（$j=k$）一个价 | 18 |
| `od` | 每个 $(j,k)$ 一个价 | 81 |

$$
z_{g\ell}\in\{0,1\},\qquad \sum_{\ell}z_{g\ell}=1,\qquad
p_{jk}=\sum_{\ell}P_\ell\,z_{g(j,k),\ell}.
\tag{PR-1}
$$

**为什么用离散菜单而不是连续价格**：(a) 第一阶段变成纯二元，integer L-shaped、Lagrangian 割、按 $z$ 做对偶分解都有精确性保证；
(b) 工资项 $p\,\tilde m$ 能精确线性化（§3），不需要 `NonConvex=2`；(c) 价格只通过 $u_{ijk}=p_{jk}-d_{ij}-c_{jk}$ 和整数阈值比较起作用，
整数菜单不丢关键取值。连续价格是可选扩展，代价是双线性项。

---

## 2. 第二阶段：每个场景一个 T 期确定性 MILP

$$
Q(p,\xi)=\max\ \sum_{t=1}^{T}\Bigl(\sum_{\ell<K}\sum_{ij}R_{ij}Y^{\ell,t}_{ij}-C_p\sum_iL^t_i-\sum_{jk}p_{jk}\tilde m^t_{jk}\Bigr)
\quad\text{s.t. EXP-BATTLVL 的全部约束，}d^t=\xi^t,\ t=1..T.
\tag{2S-2}
$$

两阶段问题（SAA，$K_s$ 个等概率场景）：

$$
v^{\mathrm{2S}}=\max_{z\ \text{满足 (PR-1)}}\ \frac1{K_s}\sum_{s=1}^{K_s}Q\bigl(p(z),\xi^s\bigr).
\tag{2S-1}
$$

第一阶段没有自己的目标项：价格的全部影响都在第二阶段（工资、谁接单、接哪单）。

---

## 3. 价格进入第二阶段的两个位置

### 3.1 工资项：精确线性化

$p_{jk}\tilde m^t_{jk}$ 是双线性的。对每个场景、每期引入 $w^t_{jk\ell}\ge0$：

$$
\sum_\ell w^t_{jk\ell}=\tilde m^t_{jk},\qquad
w^t_{jk\ell}\le Q_3\,z_{g(j,k),\ell},\qquad
\text{工资}=\sum_{jk\ell}P_\ell\,w^t_{jk\ell}.
\tag{PR-2}
$$

$z_{g\cdot}$ 恰有一个为 1（记为 $\ell^*$），于是 $w_{jk\ell}=0$（$\ell\ne\ell^*$）、$w_{jk\ell^*}=\tilde m_{jk}$，工资恰为 $P_{\ell^*}\tilde m_{jk}$。
$Q_3$ 有效，因为 $\tilde m\le M\le B_{\mathrm{total}}=Q_3$。固定价格（评估）时直接写 $p_{jk}\tilde m_{jk}$。

### 3.2 稳定匹配：效用随价格变

$u_{ijk}=p_{jk}-d_{ij}-c_{jk}$ 现在是第一阶段变量的仿射函数。(S3)、(S6) 仍是线性的：

$$
p_{jk}-d_{ij}-c_{jk}\ \ge\ s_i-Q_2(1-\eta_{ijk}),\qquad
s_i\ \ge\ p_{jk}-d_{ij}-c_{jk}-Q_2\,\delta_{ijk}.
$$

$Q_2$ 改为 $\bar P+\bar d+\bar c$，$\bar P=\max\mathcal P$（toy：$8+1+3=12$）。model_v2 §8.4 的有效性证明原样成立，只需 $0\le p\le\bar P$。

**价格的经济作用**（实现里自动体现，不需要额外约束）：

- **个体理性**：$s_i\ge0$ 加 (S3) 意味着 $i$ 的工人接 $(j,k)$ 当且仅当 $p_{jk}\ge d_{ij}+c_{jk}$。toy 中 $d\in\{0,1\}$，
  调度 $c=1$、换电 $c=3$，阈值是 $\{1,2\}$（调度）和 $\{3,4\}$（换电）。
- **工人偏好**：价格改变 $u$ 的排序，从而改变稳定匹配。
- **$u=0$ 视为接单**（乐观，与 optimistic matching 一致）。所以平台会把价格压到恰好等于某个阈值。

### 3.3 bilevel 的读法

平台（leader）定价，工人与任务之间形成稳定匹配（follower）。用稳定性约束 (S1)–(S6) 把 follower 写成约束，平台在全部稳定匹配中选最有利的那个，
这是 **optimistic bilevel** 的单层改写。pessimistic 版本需要另一套方法，留作后续。

---

## 4. 信息结构

- 非预见性只作用于 $p$：同一个价格用于全部场景。
- 第二阶段对 $\xi$ 是**完全信息**：stage $t$ 的调度可以利用 $d^{t+1},\ldots,d^T$（例如为下午的需求提前囤车）。
- 所以 $Q(p,\xi)$ 是给定价格下**能达到的最好运营**，是任何在线运营策略在同一价格下的上界。
  若以后要评估"这个价格在逐期揭示需求时值多少"，需要把 $p^*$ 放回多阶段模型（SDDiP 或滚动求解）里评估。

所有结果报告必须标明四个乐观假设：乐观短缺分配、乐观稳定匹配、平台决定哪些 OD 拿 H 车（EXP-BATTLVL）、第二阶段全天预见。

---

## 5. 删除与保留

| model_v2 / EXP-BATTLVL 中的设定 | 处理 | 说明 |
|---|---|---|
| linking state 二进制展开、bit 数、逐分量可达界 (ST-6′/ST-7′) | **删除** | 场景内各期状态是普通连续变量，由等式约束连起来 |
| local copy 与 box 约束 | **删除** | 没有 state copy |
| A2 bounded-elastic recourse、$\rho_t$、code-overflow slack | **删除** | 不需要 complete recourse；任何价格下第二阶段都可行（服务强制、发布可为 0、稳定匹配总存在） |
| initial cut $\underline\theta_t$ | **删除** | 没有割 |
| 冻结重组树、stagewise independence、确定性根节点 | **删除** | 场景是完整轨迹，见 [02](02_instances.md) |
| SDDP.jl policy graph | **删除** | 直接建 JuMP 模型 |
| Γ 缩放与欧几里得分解 $\widetilde A=\Gamma I+r$ | **保留** | 只为 floor 规则 $I=\lfloor A\rfloor$ 的精确表示；Γ 不再影响状态位数（K=2：20，K=3：10） |
| (SC-5) $D_i^t\le Q_3$ | **保留** | (D2) 的 Big-M 精确性，逐场景核验 |
| $Q_1,Q_3$，$Q_2$ 的形式 | **保留**（$Q_2$ 用 $\bar P$） | |
| 守恒 (CONS-B)/(CONS-W) | **保留** | 每个求解出的场景逐期自动检查 |
| overnight reset、零 terminal value、初始全部 1 档 | **保留** | |

**仍然存在的建模问题**：EXP-BATTLVL 已验证"期望流 + 向下取整"会让小数车滞留、丢客。两阶段 MILP 精确地计算了这部分损失（不再有割看不见它的问题），
但它仍是建模假设造成的损失，不是真实运营的损失。若要去掉，按 EXP-BATTLVL §7.3 改成每个 OD 掉整数档。

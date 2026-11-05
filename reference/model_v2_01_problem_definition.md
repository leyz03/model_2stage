# 01 — Problem Definition (v2)

状态：**working specification（2026-09-12）**

本文件是 LIME-OPR **v2** 业务 formulation 的**自包含**定义。它取代
`new model/01_problem_definition.md` 与 `new model/02_information_timing.md`；
随机需求由 [02_scenario_tree.md](02_scenario_tree.md) 定义的**固定 Poisson 场景树**给出，
SDDiP 接口由 [03_sddip_interface.md](03_sddip_interface.md) 给出。

相对 v1 的逐条改动与理由见 [00_changes.md](00_changes.md)。式号与 v1 保持一致：
**未加标记**的式子与 v1 逐字相同，标 **[v2]** 的式子被本文件修改或新增。

---

## 1. 研究目标与系统边界

有限期共享单车运营系统。平台在随机顾客需求下发布车辆移动与换电任务；工人与 open jobs
之间的分配必须满足 stable matching。发布任务的数量是当期运营决策。

任务报酬 $p_{jk}$ 是人为给定、与时间无关的**固定参数**，本版不优化价格；未来再把它提升为
上层定价决策形成 bilevel 问题。

目标是在 $T$ 个决策阶段内最大化期望运营收益：顾客骑行收入 − 未满足需求惩罚 − 实际开始
执行的工人任务报酬。

## 2. 集合与索引

- 节点集合 $\mathcal N$，$|\mathcal N|=n$，索引 $i,j,k,i'$；场景枚举索引另用 $\ell$。
- 决策阶段 $\mathcal T=\{1,\ldots,T\}$，索引 $t$。
- 运营弧集合 $\mathcal E=\mathcal N\times\mathcal N$，**含自环**；顾客骑行与工人任务共用
  这个全连通有向图。
- $x_{ijk}^t$ 的三个空间索引依次为：工人出发节点 $i$、任务起点 $j$、任务终点 $k$。
- $j=k$ 的任务是节点 $j$ 的换电任务；$j\ne k$ 是车辆移动任务。
- $i=j$ 的顾客需求是合法的闭环骑行。

$\mathcal N^2$ 或 $\mathcal N^3$ 上的每个空间索引组合都属于模型。**不得**因端点相同而从
变量、约束或场景树中删除任何组合。某个 OD-stage 没有需求，只能由场景树中该分量恒为零
表达，不得删弧。

## 3. 参数与随机量

### 3.1 固定参数

| 符号 | 含义 |
|---|---|
| $A_i^0,U_i^0,W_i^0\in\mathbb Z_{\ge0}$ | 整辆车辆与整数工人初始分布；本版假设初始全部有电，故 $U_i^0=0$ |
| $R_{ij}\ge0$ | 顾客开始骑行 $i\to j$ 时立即确认的净收入 |
| $C_p\ge0$ | 每单位未满足需求的惩罚 |
| $p_{jk}\ge0$ | 工人开始任务 $j\to k$ 时立即支付的固定报酬（**参数**，非决策） |
| $d_{ij}\in\mathbb Z_{\ge0}$ | 工人从 $i$ 到任务起点 $j$ 的时间 |
| $c_{ij}\in\mathbb Z_{\ge1}$ | 顾客骑行 $i\to j$ 的时间；任务语境下 $c_{jk}$ 是执行时间，$c_{jj}$ 是换电时间 |
| $\tau_{ijk}=d_{ij}+c_{jk}\in\mathbb Z_{\ge1}$ | 位于 $i$ 的工人执行任务 $j\to k$ 的总时间 |
| $C_{\max}=\max_{i,j}c_{ij}$ | 顾客骑行最大持续时间 |
| $H_{\max}=\max_{i,j,k}\tau_{ijk}$ | 工人任务最大持续时间 |
| $\phi_{ij}\in[0,1]$ | OD $i\to j$ 的电池失效率，以**有理数**给出 |
| $B_{\mathrm{total}},W_{\mathrm{total}}$ | 系统车辆与工人总数 |

初始资源满足

$$
\sum_{i}A_i^0=B_{\mathrm{total}},\qquad
U_i^0=0\ (i\in\mathcal N),\qquad
\sum_{i}W_i^0=W_{\mathrm{total}}.
\tag{P0}
$$

三个 Big-$M$ 常数：

$$
Q_1=W_{\mathrm{total}},\qquad
Q_2=\max_{j,k}p_{jk}+\max_{i,j}d_{ij}+\max_{i,j}c_{ij},\qquad
Q_3=\sum_i(A_i^0+U_i^0)=B_{\mathrm{total}}.
\tag{BM}
$$

$Q_1$ 界定工人数量，$Q_2$ 是稳定匹配的全局效用差上界，$Q_3$ 界定整辆车辆、任务池和
服务量；缩放车辆量的相应上界是 $\Gamma Q_3$。$Q_2$ 的有效性证明见第 8.4 节。

### 3.2 缩放因子 $\Gamma$ 由实例推导 **[v2]**

本版不再把 $\Gamma$ 固定为 $100$。设 $\phi_{ij}=a_{ij}/b_{ij}$ 为最简分数，定义

$$
\boxed{\ \Gamma:=\operatorname{lcm}\{b_{ij}:i,j\in\mathcal N\},\qquad
q_{ij}^{\phi}:=\Gamma\phi_{ij}\in\{0,1,\ldots,\Gamma\}.\ }
\tag{SC-1}
$$

于是 $q_{ij}^{\phi}$ 必然是整数，且 $\Gamma$ 是使**全部**失效分拆同时精确整数化的**最小**
正整数。$\Gamma$ 必须在状态编码之前算出，因为 bit 数依赖它。

> **为什么最小很重要。** 每个车辆分量的 bit 数是
> $\lceil\log_2(\Gamma Q_3+1)\rceil$，padding 比例是
> $1-(\Gamma Q_3+1)/2^{\lceil\log_2(\Gamma Q_3+1)\rceil}$。在 toy（$\phi\equiv0.15=3/20$）上
> $\Gamma=20$ 而不是 $100$：每个车辆分量 **8 bits 而不是 11**，padding 从 **41.4% 降到 5.9%**。
> 精确性只要求 $\Gamma\phi_{ij}\in\mathbb Z$，多出来的倍数是纯损失。

$\Gamma$ 是实例的派生常数，**不是可调参数**；实现不得接受手工传入的 $\Gamma$。

### 3.3 随机需求：固定场景树 **[v2]**

随机需求由 [02_scenario_tree.md](02_scenario_tree.md) 定义的**固定场景树**给出，本节只
记录 formulation 需要的接口：

$$
\Omega_1=\{d^{1}\},\quad q_1(d^1)=1;\qquad
\Omega_t=\{d^{t,1},\ldots,d^{t,N_b}\},\quad
q_t\!\left(d^{t,\ell}\right)=\tfrac1{N_b}\quad(t\ge2),\qquad
d^{t,\ell}\in\mathbb Z_{\ge0}^{n\times n}.
\tag{SC-2}
$$

第 1 阶段需求 $d^1$ 是**已知的实例输入**，第 1 阶段因而是论文 Algorithm 2 的确定性根节点；
不确定性只在第 $2,\ldots,T$ 阶段。

**场景树一经生成即冻结，它就是本模型的随机过程。** Poisson 只是生成配方，被记录下来
用于复现，**不进入模型定义**。各阶段的 $\Omega_t$ 独立抽取，故 stagewise independence
成立（`LinearPolicyGraph` 的前提）。

$\Omega_t$ 是 $N_b$ 个**联合原子**的集合，**不是**各 OD 边际支持的 Cartesian product。因此

$$
q_t(d^t)\ne\prod_{i,j}q_{ij}^t(d_{ij}^t)
\quad\text{——}\ \textbf{不得}\ \text{使用乘积概率公式。}
\tag{SC-3}
$$

节点总出发需求与两个派生上界：

$$
D_i^t=\sum_{j}D_{ij}^t,\qquad
\bar d_{ij}^t:=\max_{\ell}d_{ij}^{t,\ell},\qquad
\bar D_i^t:=\max_{\ell}\sum_j d_{ij}^{t,\ell}.
\tag{SC-4}
$$

注意 $\bar D_i^t$ 用的是**逐场景节点总量的最大值**，而不是 $\sum_j\bar d_{ij}^t$。在
乘积支持下二者相等；在抽样树下前者**严格更紧**，因为只有 $N_b$ 个特定联合向量会出现。

允许进入求解的实例必须满足

$$
\boxed{\ \bar D_i^t\le Q_3,\qquad i\in\mathcal N,\ t\in\mathcal T.\ }
\tag{SC-5}
$$

这是 (D2) 的 Big-$M$ 精确性条件。由于支持有限且已知，(SC-5) 是**可精确核验**的等式检查，
不是概率论断。不满足的实例必须在求解前报错，**不得**静默截断需求或改写场景树。

### 3.4 期望流约定

本版不逐辆抽样电池失效，而使用确定性期望流分解：比例 $1-\phi_{ij}$ 的顾客返回车辆进入
available flow，$\phi_{ij}$ 进入 unavailable flow。为避免连续 linking state，优化模型
不直接使用这些小数车辆量，而使用其 $\Gamma$ 倍缩放整数。

由于新服务量 $Y_{ij}^t$ 被显式限制为整数、$q_{ij}^{\phi}$ 是整数，每次新产生的返回流
严格位于 $1/\Gamma$ 网格；pipeline 之后只做**移位和整数加法**，所以网格分母**不跨阶段
增长**。这是整数缩放能够精确成立的关键，也是为什么 (D3) 的整数 OD 分配不是偏好而是
**前提**：若恢复 $Y_{ij}=Y_i\rho_{ij}(\omega)$ 的比例分摊，$\rho$ 是随机分数，网格分母
每阶段都会变，整数缩放立刻失效。

"统一整数缩放"特指所有可能因期望失效分拆而成为小数的**车辆数量**统一以 $1/\Gamma$ 辆为
一个模型单位。已经以人、整辆任务车或整数请求计数的
$W,M,\tilde m,\hat m,x,G,Y,L$ **不**乘 $\Gamma$；它们通过状态方程中的系数 $\Gamma$ 与
缩放车辆量相连。

---

## 4. 变量与域

### 4.1 连续非负变量

$$
s_i^t,\ v_{ijk}^t\in\mathbb R_{\ge0}.
$$

- $s_i^t$：节点 $i$ 工人在 matching 中的最低相关效用辅助变量。非负域同时表达 outside
  option 为零与个体理性。
- $v_{ijk}^t$：精确线性化 $\delta_{ijk}^tM_{jk}^t$ 的变量；(S1) 使其在整数 $M$ 与二元
  $\delta$ 下取 $0$ 或 $M_{jk}^t$。

### 4.2 非负整数变量

车辆期望量统一用波浪号表示其 $\Gamma$ 倍缩放值：

$$
\widetilde A_i^t=\Gamma A_i^t,\quad
\widetilde U_i^t=\Gamma U_i^t,\quad
\widetilde F_i^t=\Gamma F_i^t,\quad
\widetilde{\bar F}_i^t=\Gamma\bar F_i^t,\quad
\widetilde P_{j,r}^{A,t}=\Gamma P_{j,r}^{A,t},\quad
\widetilde P_{j,r}^{U,t}=\Gamma P_{j,r}^{U,t}.
$$

带波浪号的量是模型变量；不带的只用于说明物理单位，等于相应缩放整数除以 $\Gamma$，不是
另一套独立变量。

$$
\begin{aligned}
&\widetilde A_i^t,\widetilde U_i^t,\widetilde F_i^t,\widetilde{\bar F}_i^t,
\widetilde P_{j,r}^{A,t},\widetilde P_{j,r}^{U,t},\\
&Y_i^t,Y_{ij}^t,L_i^t,I_i^t,r_i^{A,t},W_i^t,M_{jk}^t,\tilde m_{jk}^t,x_{ijk}^t,
\hat m_{jk}^t,G_{k,r}^t
\end{aligned}
\ \in\mathbb Z_{\ge0}.
$$

| 变量 | 含义 |
|---|---|
| $\widetilde A_i^t$ | 边界 $t$ 到达处理完、需求发生前，节点 $i$ **未被任务锁定**的 available 车辆缩放量 |
| $\widetilde U_i^t$ | 同一时点未被锁定的 unavailable 车辆缩放量 |
| $\widetilde F_i^t,\widetilde{\bar F}_i^t$ | 边界 $t$ 返回的有电/没电顾客车辆缩放量 |
| $\widetilde P_{j,r}^{A,t},\widetilde P_{j,r}^{U,t}$ | 顾客 outgoing pipeline 缩放量，$r=1,\ldots,C_{\max}$ |
| $Y_i^t$ | 节点 $i$ 被服务的整数总需求 |
| $Y_{ij}^t$ | 实际服务的整数 OD 流量 |
| $L_i^t$ | 未满足的整数需求 |
| $I_i^t$ | 可用于服务的整辆库存，$I_i^t=\lfloor\widetilde A_i^t/\Gamma\rfloor$ |
| $r_i^{A,t}$ | $\widetilde A_i^t$ 除以 $\Gamma$ 的余数 |
| $W_i^t$ | matching 紧前位于节点 $i$ 的 available workers |
| $\hat m_{jk}^t$ | 本期新发布的任务；发布即锁车 |
| $M_{jk}^t$ | 本期发布以后、matching 以前的全部 open jobs |
| $x_{ijk}^t$ | 从 $i$ 出发并开始任务 $j\to k$ 的工人数 |
| $\tilde m_{jk}^t$ | 本期实际开始的任务 $j\to k$ 数量 |
| $G_{k,r}^t$ | 尚需 $r$ 个 stage 在 $k$ 完成的进行中任务数，$r=1,\ldots,H_{\max}$；每单位同时占一辆任务车和一名工人 |

$\widetilde P^t,G^t$ 是 stage $t$ 的 **outgoing pipeline**：边界既有到达已处理完毕，本期新
活动也已加入。stage $t$ 子问题以 $\widetilde P^{t-1},G^{t-1}$ 为输入，以
$\widetilde P^t,G^t$ 为输出；**不使用** $t+$、$t-$ 记号。

显式界：

$$
0\le\widetilde A_i^t,\widetilde U_i^t,\widetilde F_i^t,\widetilde{\bar F}_i^t,
\widetilde P_{j,r}^{A,t},\widetilde P_{j,r}^{U,t}\le\Gamma Q_3,
\tag{BND-B}
$$

$$
0\le Y_i^t,L_i^t,I_i^t\le Q_3,\qquad
0\le Y_{ij}^t\le D_{ij}^t,\qquad
0\le r_i^{A,t}\le\Gamma-1,
\tag{BND-Y}
$$

$$
0\le W_i^t,x_{ijk}^t\le Q_1,\qquad
0\le M_{jk}^t,\tilde m_{jk}^t,\hat m_{jk}^t\le Q_3,\qquad
0\le G_{k,r}^t\le Q_1.
\tag{BND-I}
$$

(BND-B)、(BND-I) 是**逐分量一致的安全界**，用于变量声明。跨阶段 linking state 分量另有
**逐阶段更紧**的可达性界 (ST-6′)，见 [03_sddip_interface.md](03_sddip_interface.md)；
后者只用于 bit 分配与 slack 上界，不放宽这里的变量域。 **[v2]**

### 4.3 二元变量

$$
\beta_i^t,\zeta_i^t\in\{0,1\},\qquad
\delta_{ijk}^t,\eta_{ijk}^t\in\{0,1\}.
$$

- $\beta_i^t$：线性化 $Y_i^t=\min\{I_i^t,D_i^t\}$。
- $\delta_{ijk}^t$：job-side fullness 开关；$\delta=1$ 证明任务池已被距离不大于 $d_{ij}$ 的
  工人占满，但**不要求反命题成立**。
- $\eta_{ijk}^t$：$x_{ijk}^t>0\Rightarrow\eta_{ijk}^t=1$；反向不成立。
- $\zeta_i^t$：$\zeta_i^t=1\Rightarrow$ 节点 $i$ 的 available workers 全部派出；反向不成立。

$\delta,\eta,\zeta$ 都只是单向辅助开关，**不是"当且仅当"编码**。

---

## 5. 目标函数

$$
\max\quad\mathbb E\left[\sum_{t=1}^{T}\left(
\sum_{i,j}R_{ij}Y_{ij}^t
-C_p\sum_iL_i^t
-\sum_{j,k}p_{jk}\tilde m_{jk}^t\right)\right].
\tag{OBJ}
$$

期望对 [02_scenario_tree.md](02_scenario_tree.md) 的**冻结场景树**取，即
$\mathbb E[\cdot]=\sum_{d^t\in\Omega_t}q_t(d^t)[\cdot]$ 逐阶段展开。因此 (OBJ) 的最优值是
**SAA 最优值 $v^*_{\mathcal T}$**，不是真实分布下的 $v^*$。 **[v2]**

收入与工资都在活动**开始**时结算：$R_{ij}Y_{ij}^t$ 在顾客于阶段 $t$ 骑上车时确认，
$p_{jk}\tilde m_{jk}^t$ 在工人于阶段 $t$ 出发时支付。该规则对阶段 $T$ 完全相同；活动是否
在日内完成不改变已结算的收入或成本。overnight reset 不产生新的 $R$ 或 $p$ 项。因为
$p_{jk}$ 是固定参数，目标函数线性。

---

## 6. 顾客需求与返回流

### 6.1 服务量与 floor 线性化

$$
Y_i^t=\min\left\{D_i^t,\left\lfloor A_i^t\right\rfloor\right\}.
\tag{D0}
$$

floor 是已接受的业务规则：小于一辆的 available 期望余额留在库存中，但不能单独服务一位
顾客。为在 MILP 中精确表示，先作欧几里得分解

$$
\widetilde A_i^t=\Gamma I_i^t+r_i^{A,t},\qquad
0\le r_i^{A,t}\le\Gamma-1,\qquad
I_i^t,r_i^{A,t}\in\mathbb Z_{\ge0},
\tag{D1}
$$

于是 $I_i^t=\lfloor\widetilde A_i^t/\Gamma\rfloor=\lfloor A_i^t\rfloor$。再用

$$
Y_i^t\le I_i^t,\quad
Y_i^t\le D_i^t,\quad
Y_i^t\ge I_i^t-Q_3(1-\beta_i^t),\quad
Y_i^t\ge D_i^t-Q_3\beta_i^t.
\tag{D2}
$$

**精确性。** 由 $0\le I_i^t\le Q_3$ 与 (SC-5) 的 $D_i^t\le\bar D_i^t\le Q_3$ 得
$Q_3\ge|I_i^t-D_i^t|$，故 (D2) 对每个合法状态和树中**每个**场景都是 (D0) 的精确
Big-$M$ 线性化。该结论只用物理车辆界与 (SC-5)。 **[v2：由 $\bar D_i^t$ 给出，比 v1 的
$\sum_j\max\mathcal D_{ij}^t$ 紧]**

### 6.2 平台选择的整数 OD 服务与 loss

短缺**不**按 OD 需求比例分摊。平台在观察完整 $D^t$ 后选择整数 OD 服务量：

$$
\sum_{j}Y_{ij}^t=Y_i^t,\qquad
0\le Y_{ij}^t\le D_{ij}^t,\qquad
Y_{ij}^t\in\mathbb Z_{\ge0}.
\tag{D3}
$$

由 $Y_i^t\le D_i^t=\sum_jD_{ij}^t$ 且全为非负整数，(D3) 总存在整数分配。它允许平台按
即时收入与后续车辆价值选择服务哪些 OD 请求，因此是相对比例分摊的
**optimistic platform-controlled allocation**。

$$
L_i^t=D_i^t-Y_i^t.
\tag{D4}
$$

### 6.3 顾客车辆 pipeline 与返回

初始无在途流：

$$
\widetilde P_{j,r}^{A,0}=\widetilde P_{j,r}^{U,0}=0,\qquad r=1,\ldots,C_{\max}.
\tag{D5}
$$

边界 $t$ 返回的车辆恰为上一 outgoing pipeline 的 $r=1$ bucket：

$$
\widetilde F_j^t=\widetilde P_{j,1}^{A,t-1},\qquad
\widetilde{\bar F}_j^t=\widetilde P_{j,1}^{U,t-1}.
\tag{D6}
$$

处理返回后旧 pipeline 前移一个 bucket；本期新骑行进入与其持续时间对应的 bucket：

$$
\widetilde P_{j,r}^{A,t}
=\widetilde P_{j,r+1}^{A,t-1}
+\sum_{i:\,c_{ij}=r}(\Gamma-q_{ij}^{\phi})Y_{ij}^t,
\tag{D7}
$$

$$
\widetilde P_{j,r}^{U,t}
=\widetilde P_{j,r+1}^{U,t-1}
+\sum_{i:\,c_{ij}=r}q_{ij}^{\phi}Y_{ij}^t,
\tag{D8}
$$

末端零约定

$$
\widetilde P_{j,C_{\max}+1}^{A,t-1}=\widetilde P_{j,C_{\max}+1}^{U,t-1}=0.
\tag{D9}
$$

因此 $c_{ij}=1$ 的新骑行也先进入 $\widetilde P_{j,1}^{\cdot,t}$，并在边界 $t+1$ 通过 (D6)
返回 —— 它**没有绕过** pipeline。缩放流拆分满足

$$
(\Gamma-q_{ij}^{\phi})Y_{ij}^t+q_{ij}^{\phi}Y_{ij}^t=\Gamma Y_{ij}^t,
\tag{D10}
$$

所以 pipeline 化在缩放单位中精确守恒一辆被服务车辆的 $\Gamma$ 个单位。由于 (D7)–(D8) 只
移位并加整数倍 $q_{ij}^{\phi}$，从零初始条件出发的全部
$\widetilde P^A,\widetilde P^U$ 始终为整数，物理量始终精确位于 $1/\Gamma$ 网格。

---

## 7. 任务发布、matching 与资源流

### 7.1 发布任务并立即锁车

$$
\Gamma\hat m_{jj}^t\le\widetilde U_j^t,
\tag{P1}
$$

$$
\Gamma\sum_{k\ne j}\hat m_{jk}^t\le\widetilde A_j^t-\Gamma Y_j^t.
\tag{P2}
$$

旧 open jobs 对应的车辆已在以前发布时锁定，不含在 $\widetilde A^t,\widetilde U^t$ 中；本期
只通过 $\hat m^t$ 锁定新任务的整辆车。因 $\hat m$ 是整数，(P1)–(P2) 等价于物理形式的
$\hat m_{jj}^t\le\lfloor U_j^t\rfloor$、$\sum_{k\ne j}\hat m_{jk}^t\le\lfloor A_j^t\rfloor-Y_j^t$。

### 7.2 工人任务 pipeline

$$
G_{k,r}^{0}=0,\qquad r=1,\ldots,H_{\max}.
\tag{G0}
$$

$$
G_{k,r}^{t}=G_{k,r+1}^{t-1}+\sum_{i,j:\,\tau_{ijk}=r}x_{ijk}^t,
\qquad
G_{k,H_{\max}+1}^{t-1}=0.
\tag{G1, G2}
$$

$\tau_{ijk}=1$ 的任务也先进入 $G_{k,1}^t$，再在边界 $t+1$ 完成。任务完成时，一辆任务车与
一名工人同时在终点 $k$ 恢复 available。

### 7.3 车辆状态转移

$$
\widetilde A_j^t
=\widetilde A_j^{t-1}-\Gamma Y_j^{t-1}
-\Gamma\sum_{k\ne j}\hat m_{jk}^{t-1}
+\widetilde P_{j,1}^{A,t-1}
+\Gamma G_{j,1}^{t-1},
\qquad t=2,\ldots,T,
\tag{B1}
$$

$$
\widetilde U_j^t
=\widetilde U_j^{t-1}-\Gamma\hat m_{jj}^{t-1}+\widetilde P_{j,1}^{U,t-1},
\qquad t=2,\ldots,T.
\tag{B2}
$$

任务在发布时已锁车，开始执行时**不再**从 $\widetilde A,\widetilde U$ 扣车。移动与换电任务
完成后均以一整辆车（$\Gamma G_{j,1}^{t-1}$ 个缩放单位）进入 $\widetilde A_j^t$。若初始缩放
状态为整数，(B1)–(B2) 保持整数闭合。状态转移只对 $t=2,\ldots,T$ 定义，**不引入索引超过
$T$ 的状态**。

### 7.4 工人容量与状态转移

$$
\sum_{j,k}x_{ijk}^t\le W_i^t,
\tag{W1}
$$

$$
W_k^t=W_k^{t-1}-\sum_{i,j}x_{kij}^{t-1}+G_{k,1}^{t-1},
\qquad t=2,\ldots,T.
\tag{W2}
$$

由 $c_{jk}\ge1$ 得 $\tau_{ijk}\ge1$，故 stage $t$ 出发的工人不可能在同一 stage 再次
available。

### 7.5 Open jobs 与实际开始任务

$$
M_{jk}^t=M_{jk}^{t-1}-\tilde m_{jk}^{t-1}+\hat m_{jk}^t,\qquad t=2,\ldots,T,
\tag{M1}
$$

$$
\tilde m_{jk}^t=\sum_ix_{ijk}^t,
\tag{M2}
$$

$$
\tilde m_{jk}^t\le M_{jk}^t.
\tag{M3}
$$

若初始无遗留 open jobs，则 $M_{jk}^1=\hat m_{jk}^1$。阶段 $t$ 真正传给下一阶段的
open-job 状态是 $M_{jk}^t-\tilde m_{jk}^t$；本 formulation **不**为它另设 $O$ 变量。

### 7.6 一步 Markov 接口

在 stage $t$ 观察 $D^t$ 以前，当前决策所需信息由以下量完整给出：

$$
S_t=\left(
\widetilde A^t,\ \widetilde U^t,\ W^t,\ M^{t-1}-\tilde m^{t-1},\
\left\{\widetilde P_{j,r}^{A,t-1},\widetilde P_{j,r}^{U,t-1}\right\}_{r=2}^{C_{\max}},\
\left\{G_{k,r}^{t-1}\right\}_{r=2}^{H_{\max}}
\right).
\tag{MK1}
$$

上一 pipeline 的 $r=1$ bucket 已通过 (B1)、(B2)、(W2) 计入
$\widetilde A^t,\widetilde U^t,W^t$，故不在最小 incoming state 中重复保留。给定 (MK1)、
当期 $D^t$ 与当期决策，(D7)–(D9)、(G1)、(M1) 与资源转移唯一确定下一状态。因此 (MK1)
相同但更早历史不同的两条路径拥有相同的未来可行域与条件价值。

对 $t=1$，leftover open jobs 与 pipeline tail 均取零；若 $C_{\max}=1$ 或 $H_{\max}=1$，
相应 $r\ge2$ tail 是空集合。

**linking 时点（选择 A）。** linking state 定义在下一阶段边界到达已处理完毕、下一阶段
需求尚未观察的时点。stage $t$ outgoing pipeline 在边界 $t+1$ 按互斥规则进入下一状态：

$$
\begin{aligned}
\widetilde P_{j,1}^{A,t}&\to\widetilde A_j^{t+1}, &
\widetilde P_{j,1}^{U,t}&\to\widetilde U_j^{t+1}, &
G_{k,1}^{t}&\to\left(\widetilde A_k^{t+1},W_k^{t+1}\right),\ \text{车辆贡献 }\Gamma G_{k,1}^t,\\
\widetilde P_{j,r}^{A,t}&\to\widetilde P_{j,r-1}^{A,t+1}, &
\widetilde P_{j,r}^{U,t}&\to\widetilde P_{j,r-1}^{U,t+1}, &
G_{k,r}^{t}&\to G_{k,r-1}^{t+1},\qquad r\ge2,
\end{aligned}
\tag{MK2}
$$

其中 $r\ge2$ 的箭头只表示**旧 pipeline 的移位（carryover）**；stage $t+1$ 的新骑行与新
任务还要按 (D7)–(D8)、(G1) 加入对应 bucket。

> 需要区分两件事：完整 formulation 本身从不重复计算，因为 (B1)–(B2)、(W2) 只使用一次旧
> $r=1$ bucket，而 (D7)–(D8)、(G1) 的移位只读取旧 $r+1$ bucket。只传递 $r\ge2$ **不是**
> 修复错误，而是因为在所选 linking 时点 $r=1$ 已转化为 $\widetilde A,\widetilde U,W$，
> 不再属于"仍在途"的最小状态。

$\widetilde F^t,\widetilde{\bar F}^t$ 由 (D6) 完全决定，不是 $S_t$ 的独立分量，也不出现在
stage $t$ 之后的任何约束中，因此从 post-arrival 单阶段接口中消去不丢失信息。

---

## 8. Stable matching

### 8.1 偏好与 blocking pair

工人从节点 $i$ 执行任务 $j\to k$ 的效用为固定常数

$$
u_{ijk}=p_{jk}-d_{ij}-c_{jk}.
$$

同一节点的工人同质，同一 OD 类型的任务同质。工人偏好更高的 $u_{ijk}$；任务偏好距任务
起点 $j$ 更近的工人，相同距离视为无严格 job-side 偏好。

formulation 保留完整索引范围 $i,j,k\in\mathcal N$，**不**定义或排除"不可接受组合"。非负
变量 $s_i^t$ 表达 outside option 为零；被匹配组合的个体理性由 (S2)–(S3) 共同约束。

### 8.2 Job-side fullness 的精确线性化

令 $H_{ijk}^t=\sum_{i':\,d_{i'j}\le d_{ij}}x_{i'jk}^t$。以下约束精确线性化
$v_{ijk}^t=\delta_{ijk}^tM_{jk}^t$ 并强制 $H_{ijk}^t\ge v_{ijk}^t$：

$$
v_{ijk}^t\le H_{ijk}^t,\quad
v_{ijk}^t\le M_{jk}^t,\quad
v_{ijk}^t\le Q_3\delta_{ijk}^t,\quad
v_{ijk}^t\ge M_{jk}^t-Q_3(1-\delta_{ijk}^t),\quad
v_{ijk}^t\ge0.
\tag{S1}
$$

因每个 open job 锁定一辆车，$M_{jk}^t\le B_{\mathrm{total}}=Q_3$，故 (S1) 直接使用
$Q_3$。**这里必须是 $Q_3$ 而不是 $Q_1$** —— 任务池可以超过工人总数，用 $Q_1$ 会得到
错误的 fullness 逻辑。

### 8.3 Worker-side 与 stability 约束

$$
x_{ijk}^t\le Q_1\eta_{ijk}^t,
\tag{S2}
$$

$$
u_{ijk}\ge s_i^t-Q_2(1-\eta_{ijk}^t),
\tag{S3}
$$

$$
\sum_{j,k}x_{ijk}^t\ge W_i^t-Q_1(1-\zeta_i^t),
\tag{S4}
$$

$$
s_i^t\le Q_2\zeta_i^t,
\tag{S5}
$$

$$
s_i^t\ge u_{ijk}-Q_2\delta_{ijk}^t.
\tag{S6}
$$

### 8.4 $Q_2$ 的有效性

记 $\bar p=\max_{j,k}p_{jk}$、$\bar d=\max_{i,j}d_{ij}$、$\bar c=\max_{i,j}c_{ij}$，则
$Q_2=\bar p+\bar d+\bar c$ 且

$$
-\bar d-\bar c\le u_{ijk}=p_{jk}-d_{ij}-c_{jk}\le\bar p.
\tag{S7}
$$

等价性证明的辅助构造给出 $0\le s_i^t\le\bar p$。于是当 $\eta_{ijk}^t=0$ 时

$$
s_i^t-Q_2\le\bar p-(\bar p+\bar d+\bar c)=-\bar d-\bar c\le u_{ijk},
$$

(S3) 被完全关闭；当 $\delta_{ijk}^t=1$ 时

$$
u_{ijk}-Q_2\le\bar p-(\bar p+\bar d+\bar c)\le0\le s_i^t,
$$

(S6) 也被完全关闭。因此该 $Q_2$ 对完整 $i,j,k$ 索引范围（**含净效用为负的组合**）均有效。

### 8.5 多重稳定匹配的选择

模型可能存在多个满足 stability 的 matching。本版采用 **optimistic stable matching**：
平台在所有满足 (M2)–(M3)、(W1) 和 (S1)–(S6) 的 stable matchings 中，联合选择使平台
目标最有利的 $x$。

> **这是一个乐观假设，与 (D3) 的 optimistic allocation 叠加。** 报出的收益是"平台最优
> 稳定匹配 + 平台最优短缺分配"下的值，**系统性高于**工人自主选择时的真实收益。任何
> 结果报告必须标明这一点；pessimistic 对照留作后续工作。

---

## 9. 事件时序与信息结构

### 9.1 单个 stage 的事件顺序

```text
stage t starts at boundary t
│
├─ 处理以前骑出、现在到期的顾客车辆
│    ├─ 有电返回（缩放量）: F̃_j^t = P̃^A_{j,1}(t-1)
│    └─ 没电返回（缩放量）: F̃̄_j^t = P̃^U_{j,1}(t-1)
│
├─ 处理以前开始、现在完成的工人任务  G_{k,1}(t-1)
│    ├─ 任务车辆在终点恢复 available
│    └─ 工人在终点恢复 available
│
├─ 确定 Ã_i^t、Ũ_i^t、W_i^t                    ← linking state 时点（选择 A）
│
├─ 观察完整 OD 需求 d^t ∈ Ω_t，计算 D_i^t = Σ_j d_ij^t
│
├─ 确定整辆可服务库存 I_i^t = ⌊Ã_i^t/Γ⌋
│
├─ 顾客骑走 Y_i^t = min{I_i^t, D_i^t}
│    ├─ 平台在 0 ≤ Y_ij ≤ d_ij^t、Σ_j Y_ij = Y_i 中选择整数 OD 分配
│    └─ 骑行收入 R_ij 在出发时立即确认
│
├─ 平台发布移动与换电任务 m̂_jk^t              ← 发布即锁车
│
├─ 确定 matching 紧前的 open-job pool M_jk^t   ← 含遗留任务 + 本期新任务
│
├─ W_i^t 与 M_jk^t 之间进行 stable matching
│    └─ 得到 x_ijk^t 与 m̃_jk^t（多重稳定匹配时取 optimistic）
│
├─ 匹配的工人立即出发并执行任务
│    └─ 任务报酬 p_jk 在出发时立即支付
│
└─ stage t ends; time advances to boundary t+1
```

### 9.2 非预见性

在阶段 $t$ 决策时，平台可以使用：截至边界 $t$ 的全部历史、当期
$\widetilde A^t,\widetilde U^t,W^t$、以前遗留的 open jobs、当期完整 $d^t$。平台**不能**
使用任何未来需求。相同可观察历史上的决策必须相同。形式化地，令
$\mathcal F_t=\sigma(D^1,\ldots,D^t)$，则阶段 $t$ 的服务、发布与 matching 决策必须对
$\mathcal F_t$ 可测。

由于 $\Omega_1=\{d^1\}$ 是单点，$\mathcal F_1$ 是平凡的：第 1 阶段的决策是确定性的，
stage-1 子问题就是 SDDiP 的确定性根节点（见 [03_sddip_interface.md](03_sddip_interface.md) §5）。

由于场景树各阶段独立抽取，

$$
\Pr(D^{t+1}=d\mid\mathcal F_t)=\Pr(D^{t+1}=d),
$$

但各阶段的 $\Omega_t$ 可以不同，故只要求 stagewise independence，**不要求同分布**。

### 9.3 返回与完成时刻

- 顾客在阶段 $r$ 骑出 $i\to j$，骑行时间 $c_{ij}$，在边界 $r+c_{ij}$ 返回。
- 工人在阶段 $r$ 从 $i$ 开始任务 $j\to k$，总时间 $d_{ij}+c_{jk}$，工人与任务车在边界
  $r+d_{ij}+c_{jk}$ 到达 $k$。
- 任何在阶段 $t$ 开始的任务都不能在同一阶段再次 available。

所有活动统一通过剩余时间 pipeline 表示；持续时间恰为一个 stage 的活动进入 $r=1$，在
下一边界释放，**不需要单独的直接到达公式**。

---

## 10. 初始条件与边界

### 10.1 左边界与阶段 1

$$
\widetilde P_{j,r}^{A,0}=\widetilde P_{j,r}^{U,0}=0,\qquad
G_{k,r}^{0}=0,
\tag{I1}
$$

$$
\widetilde A_i^1=\Gamma A_i^0,\qquad
\widetilde U_i^1=\Gamma U_i^0=0,\qquad
W_i^1=W_i^0,\qquad
M_{jk}^1=\hat m_{jk}^1.
\tag{I2}
$$

特别地，$M^1$ **不是**预先设为零的初始参数。阶段 1 先观察 $d^1$、服务顾客并决策
$\hat m^1$，然后由 $M^1=\hat m^1$ 形成第一期 matching 使用的任务池。

### 10.2 右边界与 overnight reset

阶段 $T$ 与其他阶段完全相同：顾客可以开始骑行，平台可以发布任务，工人可以匹配并开始
任务；相应收入与成本按第 5 节在阶段 $T$ 立即结算。骑行或任务**不需要**在阶段 $T$ 内完成。

有限期 formulation 只定义 $t\in\mathcal T$ 的决策与状态，状态转移只写到阶段 $T$，
**不引入**任何显式 end-of-day 变量、额外 closing stage、索引为 $T+1$ 的状态或末期活动
禁令。阶段 $T$ 以后仍在进行的骑行与任务在现实的 overnight interval 中继续并全部完成；
尚未开始的 open jobs 也由外生夜间运营统一完成，而不是在日内决策中取消。随后车辆完成
充电与重布局，工人恢复固定初始分布，所有 open-job 与 pipeline 记录被清除，次日重新从
(I2) 开始。

overnight 运营的工资视为与日内决策无关的固定成本。因此最后一个决策阶段之后没有
continuation reward 或 cost，算法上的 terminal continuation value 归一化为零。这个 reset
是有限期模型的**边界语义**，不是变量、约束或可被平台操纵的决策。

---

## 11. 守恒关系（实现必须验证）

在 stage $t$ 的骑行、发布和任务开始全部发生以后，以缩放单位计：

$$
\begin{aligned}
\Gamma B_{\mathrm{total}}
={}&\sum_i\left(\widetilde A_i^t-\Gamma Y_i^t-\Gamma\sum_{k\ne i}\hat m_{ik}^t
+\widetilde U_i^t-\Gamma\hat m_{ii}^t\right)\\
&+\sum_j\sum_{r=1}^{C_{\max}}\left(\widetilde P_{j,r}^{A,t}+\widetilde P_{j,r}^{U,t}\right)
+\Gamma\sum_{j,k}\left(M_{jk}^t-\tilde m_{jk}^t\right)
+\Gamma\sum_k\sum_{r=1}^{H_{\max}}G_{k,r}^t,
\end{aligned}
\tag{CONS-B}
$$

$$
W_{\mathrm{total}}=\sum_i\left(W_i^t-\sum_{j,k}x_{ijk}^t\right)
+\sum_k\sum_{r=1}^{H_{\max}}G_{k,r}^t.
\tag{CONS-W}
$$

这两式是由初始守恒、(D7)–(D8)、(G1)、(B1)–(B2)、(W2)、(M1)–(M3) **推导出的 invariant**，
不是额外创造资源的自由约束；实现时必须作为自动测试逐阶段检查。

---

## 12. 代码契约

- 顾客骑行弧与任务弧必须都枚举 $\mathcal N\times\mathcal N$；**不得**过滤 $i=j$ 的闭环
  骑行或 $j=k$ 的换电任务。零需求只能由场景树中该分量恒为零表达。
- $\Gamma$ 必须由 (SC-1) 从 $\phi$ 推导，**不得**接受手工传入；$q_{ij}^{\phi}$ 必须是
  整数，且**不得**用浮点乘法生成顾客返回流。
- $\widetilde A,\widetilde U,\widetilde F,\widetilde{\bar F},\widetilde P^A,\widetilde P^U$
  必须始终以缩放整数存储；只有报告物理车辆量时才除以 $\Gamma$。
- stage noise 必须严格从冻结场景树读取，**不得**在运行时重新抽样、拟合分布或假定 Poisson。
- **不得**使用乘积概率公式 (SC-3)；概率必须取自树。
- 每个实例必须核验 (SC-5)；否则 (D2) 的 Big-$M$ 线性化不得投入求解。
- OD 服务必须满足 $Y_{ij}^t\in\mathbb Z_+$、$Y_{ij}^t\le d_{ij}^t$、
  $\sum_jY_{ij}^t=Y_i^t$；**不得**恢复为按 $d_{ij}^t/D_i^t$ 的比例分摊，也不得保留任何
  $\rho$ 变量。
- outgoing state **不得**依赖未来需求或未来决策。
- matching 使用的 $W^t,M^t$ 必须是 matching 紧前快照。
- 新任务发布时锁车，任务开始时**不得**重复扣车。
- pipeline 递推为 $\widetilde P^{t-1}\to\widetilde P^t$、$G^{t-1}\to G^t$；linking interface
  采用选择 A，传递已吸收 $r=1$ 到达的下一期资源与父 pipeline 的 $r\ge2$ 尾部，**不使用**
  含义重叠的 $t+$、$t-$ 变量。
- 所有 duration-one 活动必须先进入 $r=1$ bucket，再在下一边界释放。
- (S1) 的 Big-$M$ 必须是 $Q_3$，(S2)/(S4) 是 $Q_1$，(S3)/(S5)/(S6) 是 $Q_2$。任何
  混用必须使测试失败。
- $\delta,\eta,\zeta$ 只按单向含义测试，**不得**误测为"当且仅当"。
- 最后一个决策阶段为 $T$；**不得**创建显式 end-of-day 状态、额外 stage 或 continuation cut。
- pipeline shift 与 (CONS-B)/(CONS-W) 必须有逐阶段自动测试。
- 任何结果报告必须标明目标值是 $v^*_{\mathcal T}$（SAA 最优值）而非 $v^*$，且策略处于
  optimistic allocation + optimistic stable matching 之下。

# 电池分档模型：K 档电量（H/M/L）对 model_v2 的改动

状态：**实验性扩展（2026-10-01）**，以 [model_v2/01_problem_definition.md](../../model_v2/01_problem_definition.md)
为底稿。没有提到的部分（matching、工人、任务池、时序、overnight reset）与 v2 一字不差。
实现在 [battery_levels.jl](battery_levels.jl)。

## 1. 档位与假设

- 档位 $\ell=1,\ldots,K$，$\ell=1$ 电量最高。$\ell=1..K-1$ 可租；$\ell=K$ 就是 v2 的
  "unavailable"，只能靠换电任务离开。$K=3$ 时 $1,2,3=$ H, M, L。
- **$K=2$ 就是 v2**：唯一可租档 = $A$，$K$ 档 = $U$。代码在 $K=2$ 上与
  `model_v2.jl` 的 state bits、变量数、约束数逐项相等，20 轮训练的 bound 和仿真结果也逐位相同。
- **用户偏好：优先骑高电量（H-first）。** 节点有 H 时不会有人骑 M。
- **哪些 OD 拿到 H、哪些拿到 M 仍由平台选**（整数、乐观，与 (D3) 一致）。H-first 只
  固定每档的**数量**，不固定 OD。
- **换电**：$K$ 档 → 1 档。
- **调度（move）只搬 1 档车**（`move=:top`），完成后车仍是 1 档。

## 2. 档位转移概率

每档内电量均匀分布。死档 $K$ = $[0,\theta_L)$，可用区间 $[\theta_L,1]$ 等分成 $K-1$ 档，
每档宽 $w=(1-\theta_L)/(K-1)$。OD $i\to j$ 一次骑行耗电 $\Delta_{ij}$。从 $\ell$ 档出发、到达
$\ell'$ 档的概率 $\pi_{ij}(\ell\to\ell')$ = 区间 $[a-\Delta,\,a+w-\Delta]$ 落进 $\ell'$ 档的比例
（$[a,a+w]$ 是 $\ell$ 档）。$\Delta\le w$ 时就是

$$
\pi_{ij}(\ell\to\ell)=1-\Delta_{ij}/w,\qquad \pi_{ij}(\ell\to\ell+1)=\Delta_{ij}/w .
$$

**标定**：v2 的 $\phi_{ij}$ 解释成 $\Delta_{ij}/(1-\theta_L)$，取 $\theta_L=0.2$。这样**每辆车的物理
耗电量与 $K$ 无关**，只是档切得更细，所以不同 $K$ 的比较是公平的。toy 的 $\phi=0.15$ 给出
$\Delta=0.12$：

| $K$ | 掉档概率 $\Delta/w$ | $\Gamma$ |
|---|---|---|
| 2 (v2) | 3/20 | 20 |
| 3 | 3/10 | 10 |
| 4 | 9/20 | 20 |
| 5 | 3/5 | 5 |

巴黎实例的 $\phi_{ij}=0.05+0.01\cdot\text{km}$ 不是短分数，所有概率先四舍五入到 $1/20$ 网格
（最大余数法，行和仍为 1），再按 (SC-1) 取 $\Gamma$。

缩放整数 $q_{ij}^{\ell\ell'}=\Gamma\,\pi_{ij}(\ell\to\ell')$，$\sum_{\ell'}q^{\ell\ell'}_{ij}=\Gamma$。
$\Gamma$ 取全部 $\pi$ 分母的 lcm；$Y$ 是整数，所以 pipeline 仍然待在 $1/\Gamma$ 网格上，v2 的
精确缩放论证照搬。

## 3. 改动的式子

**状态**：$\widetilde A^{\ell}_j\ (\ell<K)$、$\widetilde U_j$、pipeline $\widetilde P^{\ell'}_{j,r}\ (\ell'=1..K)$；
其余同 v2。

**向下取整**，每个可用档一份 (D1)：
$\widetilde A^{\ell}_i=\Gamma I^{\ell}_i+r^{\ell}_i,\ 0\le r^{\ell}_i\le\Gamma-1$。
半辆 H 加半辆 M **不能**凑成一辆可租车，这是 v2 floor 规则的直接推广。

**H-first 服务**：对每个 $\ell<K$，令 $Y^{\le\ell}_i$ 为由 $1..\ell$ 档服务的人数，

$$
Y^{\le\ell}_i=\min\Bigl\{\textstyle\sum_{\ell'\le\ell}I^{\ell'}_i,\ D_i\Bigr\}\quad\text{（每个 }(\ell,i)\text{ 一个 }\beta^{\ell}_i\text{，同 (D2) 的 big-}M\text{）},
$$

$Y^{\ell}_i=Y^{\le\ell}_i-Y^{\le\ell-1}_i$。可以验证 $Y^{\ell}_i=\min\{I^{\ell}_i,\,D_i-Y^{\le\ell-1}_i\}$，正是
"先 H 后 M"。$Q_3$ 仍然够做 big-$M$，因为 $\sum_\ell I^{\ell}_i\le B_{\text{total}}=Q_3$。

**OD 分配**：$\sum_jY^{\ell}_{ij}=Y^{\ell}_i$，$\sum_\ell Y^{\ell}_{ij}\le d_{ij}$，$L_i=D_i-Y^{\le K-1}_i$。

### 3.1 "哪些 OD 拿到 H 车由平台决定"的具体实现

代码：[battery_levels.jl:334-361](battery_levels.jl#L334-L361)。服务分两层变量：

| 变量 | 含义 | 由谁决定 |
|---|---|---|
| `Ycum[ℓ,i]` $=Y^{\le\ell}_i$ | 节点 $i$ 由 $1..\ell$ 档服务的总人数 | **H-first 规则强制**，平台不能选 |
| `Yod[ℓ,i,j]` $=Y^{\ell}_{ij}$ | 用 $\ell$ 档车服务的 $i\to j$ 乘客数 | **平台在下面两条约束内自由选** |

**第一层（每档服务多少人）被 H-first 定死**：第 355–358 行用 big-$M$ + 二元 $\beta^\ell_i$ 精确实现
$Y^{\le\ell}_i=\min\{\sum_{\ell'\le\ell}I^{\ell'}_i,D_i\}$。以 $K=3$ 为例：H 服务人数 = min(H 车数, 需求)，
H+M 服务人数 = min(H+M 车数, 需求)，两者相减就是 M 服务人数。"有 H 就没人骑 M"是硬约束。

**第二层（这些人分到哪些 OD）只受两条约束**：

- 第 359 行 $\sum_jY^{\ell}_{ij}=Y^{\ell}_i$：每档的人必须分完；
- 第 360 行 $\sum_\ell Y^{\ell}_{ij}\le d_{ij}$：每个 OD 的总服务不超过该 OD 的需求。

除此之外**没有任何约束规定哪个 OD 拿哪一档**，所以优化器（= 平台）会选对目标最有利的分法。收入 $R_{ij}$ 与档位无关
（目标函数第 441 行对 $\ell$ 求和），档位只通过转移系数 $q^{\ell\ell'}_{ij}$ 影响未来：长途 OD 的 $\Delta_{ij}$ 大、掉档概率高，
所以平台会**把 H 车派给长途、把 M 车派给短途**，让更少的车掉到 L。

**为什么这样设**：

1. 与 v2 的 (D3) 一致——v2 在缺车时也由平台选择服务哪些 OD（optimistic allocation）。
2. 按需求比例分（$Y^{H}_{ij}=Y^{H}_i\cdot d_{ij}/D_i$）会使 $Y^{\ell}_{ij}$ 不是整数，pipeline 离开 $1/\Gamma$ 网格，整数缩放失效；
   v2 禁止比例分摊 $\rho$ 用的是同一个理由。

**这是乐观假设**：现实中骑长途的人不一定拿到满电车，模型给出的是平台"能做到的上限"。若要更现实，可以加
"各 OD 的 H 份额与比例份额相差不超过 1"的约束：
$\bigl|Y^{H}_{ij}-Y^{H}_i\,d_{ij}/D_i\bigr|<1$。它对决策变量是线性的，系数 $d_{ij}/D_i$ 随场景变化，可在 SDDP.jl 的
`parameterize` 里用 `set_normalized_coefficient` 逐场景设置。**目前未实现。**

**发布**：$\Gamma\hat m_{jj}\le\widetilde U_j$；
$\Gamma\sum_{k\ne j}\hat m_{jk}\le\widetilde A^{1}_j-\Gamma Y^{1}_j$。

**pipeline**：

$$
\widetilde P^{\ell'}_{j,r}=\widetilde P^{\ell',\text{prev}}_{j,r+1}+\sum_{i:c_{ij}=r}\sum_{\ell<K}q^{\ell\ell'}_{ij}Y^{\ell}_{ij}.
$$

**转移**：

$$
\begin{aligned}
\widetilde A^{1}_j{}' &= \widetilde A^{1}_j-\Gamma Y^{1}_j-\Gamma\textstyle\sum_{k\ne j}\hat m_{jk}+\widetilde P^{1}_{j,1}+\Gamma G_{j,1},\\
\widetilde A^{\ell}_j{}' &= \widetilde A^{\ell}_j-\Gamma Y^{\ell}_j+\widetilde P^{\ell}_{j,1}\qquad(1<\ell<K),\\
\widetilde U_j{}' &= \widetilde U_j-\Gamma\hat m_{jj}+\widetilde P^{K}_{j,1}.
\end{aligned}
$$

## 4. 状态界（ST-6′ 的推广）

- $\widetilde A^{\ell}$：$\Gamma Q_3$。
- $\widetilde U_j$：$\Gamma U^0_j+\sum_{s<t}\sum_i\max_\ell q^{\ell K}_{ij}\,\bar d^s_{ij}$（与 $\Gamma Q_3$ 取小）。
- $\widetilde P^{\ell'}_{j,r}$：v2 的展开界，系数换成 $\max_\ell q^{\ell\ell'}_{ij}$。之所以成立，是因为
  $\sum_\ell Y^{\ell}_{ij}\le\bar d_{ij}$。系数恒为 0 的分量直接删掉。
- $W, G, M-\tilde m$：同 v2。

## 5. 守恒（每个仿真状态都检查）

$$
\Gamma B_{\text{total}}=\sum_{\ell<K,j}\widetilde A^{\ell}_j+\sum_j\widetilde U_j+\sum_{\ell',j,r\ge2}\widetilde P^{\ell'}_{j,r}
+\Gamma\sum(M-\tilde m)+\Gamma\sum_{r\ge2}G,\qquad
W_{\text{total}}=\sum W+\sum_{r\ge2}G .
$$

## 6. 这里做了、需要在报告中说明的乐观假设

v2 已有两个：乐观的短缺分配与乐观的稳定匹配。本扩展再加一个：
平台决定哪些 OD 拿到 H、哪些拿到 M，所以会把 H 派给长途（实现与可选的现实化约束见 §3.1）。
如果这一条改成按比例分配，pipeline 会离开 $1/\Gamma$ 网格，整数缩放就失效了，原因与 v2 禁止 $\rho$ 相同。

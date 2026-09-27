# 单步动作均值标准化 GRPO baseline：算法草案

> 状态：已确认的对比方法记录。Game24 实现和 CPU 校验已完成，实际 4B 模型的 A100 训练尚未进行。本方法的训练优势只使用当前步骤 \(a\) 的动作分组统计，不使用下一步 \(a'\) 的门控修正。策略更新沿用[两步方案](./two_step_gated_grpo_theory.md)已确定的逐 token 方式，并启用相同的逐 token KL。Game24 的完整参数及 INVALID 特例见[实验协议](./game24_experiment_protocol_and_logging.md)与[配置](./game24_experiment/experiment.json)。

## 1. 与两步方案的关系

**已确定并沿用的内容：**一次编辑或工具调用为一步；使用旧策略采样完整轨迹及其终局奖励；在同一完整决策状态下，先对每种不同的下一步动作分别平均终局奖励，再让不同动作的均值等权参与标准化；一步的全部模型生成 token 共用该步优势，每个 token 单独计算策略概率比并 clipping；先在每条轨迹的每一步内按该步 token 数量平均，再按步骤与轨迹平均。工具结果、环境观察和提示词只作为上下文，不作为带该步优势的预测目标。

**已确定的 baseline 差异：**每一步直接用动作均值标准化优势训练。前 20% 近零优势筛选仅属于原两步门控方案，**本 baseline 不做这一筛选，也不加入 \(\lambda\)**。因此不使用子状态 \((s,a)\) 上的 \(a'\) 统计、五种不同 \(a'\) 的门槛、\(\Delta\) 门、\(A_2\)、\(\beta\)、第二步动作筛选或相邻两步链重叠公式。baseline 的步级优势始终直接取 \(A^{\mathrm{state}}\)。

## 2. 轨迹、状态与动作分组

从相同任务与初始条件下，用旧策略 \(\pi_{\mathrm{old}}\) 收集 \(N\) 条完整轨迹 \(\tau_i\)，每条轨迹有 \(T_i\) 个决策步骤及终局奖励 \(R_i\)。状态 \(s_{i,j}\) 是轨迹 \(i\) 第 \(j\) 步的完整决策状态，\(a_{i,j}\) 是该步的编辑或工具调用动作。状态相同的判定沿用两步方案，不能为 baseline 单独放宽或收紧。

为明确每个决策位置，记同一状态 \(s\) 下观察到的步骤集合及不同动作为

\[
\mathcal I(s)=\{(i,j):s_{i,j}=s\},\qquad
\mathcal U(s)=\{a_{i,j}:(i,j)\in\mathcal I(s)\},\qquad
d_s=|\mathcal U(s)|.
\]

对于每个 \(u\in\mathcal U(s)\)，先把选择 \(u\) 的步骤归为一类，计算该类轨迹的平均**终局**奖励：

\[
\mathcal I(s,u)=\{(i,j)\in\mathcal I(s):a_{i,j}=u\},\qquad
n_s(u)=|\mathcal I(s,u)|,\qquad
q_s(u)=\frac{1}{n_s(u)}\sum_{(i,j)\in\mathcal I(s,u)}R_i.
\]

这一步在同动作内部按观察到的步骤求平均，不直接把每条轨迹的终奖投入跨动作标准差。**已确认按每次访问计数：**同一轨迹若多次访问完全相同的状态，每次访问都作为一条决策记录；若这些访问选择同一动作，\(q_s(u)\) 的求和中会重复出现该轨迹的终局奖励 \(R_i\)。这表示按访问次数加权，并不意味着这些访问在统计上相互独立。

## 3. 动作均值等权标准化与步级优势

先让状态 \(s\) 下的 \(d_s\) 个**不同动作**各占一份权重，计算动作均值的总体均值与总体标准差：

\[
\bar q_s=\frac{1}{d_s}\sum_{u\in\mathcal U(s)}q_s(u),\qquad
\sigma_{q,s}=\sqrt{\frac{1}{d_s}\sum_{u\in\mathcal U(s)}\bigl(q_s(u)-\bar q_s\bigr)^2}.
\]

再把对应动作的标准化优势回填给每个观察到的步骤：

\[
\boxed{
A^{\mathrm{base}}_{i,j}=A^{\mathrm{state}}_{i,j}=
\begin{cases}
\dfrac{q_{s_{i,j}}(a_{i,j})-\bar q_{s_{i,j}}}{\sigma_{q,s_{i,j}}},
&\sigma_{q,s_{i,j}}>0,\\[6pt]
0,&\sigma_{q,s_{i,j}}=0.
\end{cases}}
\]

同状态、同动作的步骤共享优势，即使它们所属轨迹的终局奖励不同。若仅有一种不同动作，或各动作的平均终奖相同，\(\sigma_{q,s}=0\)，该步骤的优势为 0；不因而丢弃整条轨迹。标准化时不同动作等权；策略更新时仍保留每个采样步骤，不额外乘 \(1/n_s(u)\) 做重复动作次数校正。这些规则与两步方案的 \(A^{\mathrm{state}}\) 一致。

## 4. 广播到 token 并更新策略

设第 \((i,j)\) 步由模型生成 \(L_{i,j}\) 个 token \(y_{i,j,1:L_{i,j}}\)，其中包括该步的思考文本与编辑／调用内容。全部这些 token 使用同一个步级优势：

\[
A^{\mathrm{base}}_{i,j,k}=A^{\mathrm{base}}_{i,j},\qquad k=1,\ldots,L_{i,j}.
\]

令 \(h_{i,j,k}\) 为预测该 token 时的完整上下文。逐 token 概率比与 clipped 项分别为

\[
r_{i,j,k}(\theta)=
\frac{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}
{\pi_{\mathrm{old}}(y_{i,j,k}\mid h_{i,j,k})},\qquad
\ell_{\mathrm{clip}}(r,A)=
\min\!\left\{rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\right\}.
\]

只把模型生成的 token 计入该步的 \(L_{i,j}\) 与求和。与两步方案一样，**启用逐 token KL**，使用相同的参考策略 \(\pi_{\mathrm{ref}}\) 与系数 \(\kappa>0\)。采用[原始 GRPO](https://arxiv.org/html/2402.03300)的逐 token KL 估计形式：

\[
D^{\mathrm{KL}}_{i,j,k}(\theta)=
\frac{\pi_{\mathrm{ref}}(y_{i,j,k}\mid h_{i,j,k})}
{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}
-\log\frac{\pi_{\mathrm{ref}}(y_{i,j,k}\mid h_{i,j,k})}
{\pi_\theta(y_{i,j,k}\mid h_{i,j,k})}-1.
\]

在每步内对策略 clipped 项和 KL 项一起按该步 token 数量平均，再按步骤和轨迹平均。baseline 的目标为

\[
\boxed{
J^{\mathrm{base}}(\theta)=
\mathbb E_{\{\tau_i\}\sim\pi_{\mathrm{old}}}
\left[
\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}\sum_{j=1}^{T_i}
\frac{1}{L_{i,j}}\sum_{k=1}^{L_{i,j}}
\left(
\ell_{\mathrm{clip}}\!\left(r_{i,j,k}(\theta),A^{\mathrm{base}}_{i,j}\right)
-\kappa D^{\mathrm{KL}}_{i,j,k}(\theta)
\right)
\right].}
\]

最大化 \(J^{\mathrm{base}}\)，或最小化其相反数，通过当前策略的 token 概率比和逐 token KL 项更新参数。优势与分组统计量由旧轨迹计算，在本次更新中作为固定系数。**启用逐 token KL 已确定**；Game24 首轮取 \(\kappa=0.01\)、\(\epsilon=0.2\)，参考策略为冻结的初始模型。两步方案与 baseline 使用同一参考策略、KL 定义、系数及归约方式。

## 5. 计算流程与比较边界

1. 用与两步方案相同的旧策略、任务、采样预算和终局奖励定义收集轨迹。
2. 对每个相同完整状态 \(s\)，按当前步动作 \(u\) 分组，并计算各组平均终奖 \(q_s(u)\)。
3. 让不同动作均值等权，计算 \(\bar q_s\)、\(\sigma_{q,s}\)，将 \(A^{\mathrm{base}}_{i,j}\) 回填给所有步骤。
4. 将步级优势广播给该步所有模型生成的 token；各 token 分别计算概率比和 clipping。
5. 每个模型生成 token 同时计算 clipped 项和逐 token KL 项，先在每条轨迹的每一步内按生成 token 数量平均，再按步骤和轨迹平均，更新策略。

这里的“不考虑 \(a'\)”是指**不显式按下一步动作或子状态做任何分组、筛选、门控和优势修正**。由于 \(q_s(u)\) 仍来自完整轨迹的终局奖励 \(R_i\)，奖励不可避免地间接受到后续行为影响；如果要求信号完全不含后续结果，就需要更换奖励定义，那将是另一个 baseline，不能声称沿用当前方案。

作为对比，两步方案与本 baseline 应共享终奖、状态匹配、轨迹采样、token 范围、clipping、KL、loss 归约和评估规则。baseline 只使用每一步的 \(A^{\mathrm{state}}\)；两步方案在此基础上额外使用近零筛选、\(a'\) 门控与优势修正。

## 6. 任务实例的参数与实现状态

- Game24 的参考策略、系数、数据划分、采样预算、失败处理、停止条件和评估指标已在专项协议与配置中确定。baseline 可以对已有轨迹保存两步规则的诊断值，但实际更新只取 \(A^{\mathrm{state}}\)，日志明确区分计算值与实际作用值。推广到其他任务时，仍须为该任务确认这些设置。

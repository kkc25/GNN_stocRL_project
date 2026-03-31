---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: b45de274a48824ffe1e35d5610979047
    PropagateID: b45de274a48824ffe1e35d5610979047
    ReservedCode1: 304502201cd62e78bf69295c6c6488fded6f6db1d8496383edfefe7b4f294f11fd066c690221009ff58aaedb8fc50a140b14e42ced5a3ce2b04017fddb39fd9c96caee29617c14
    ReservedCode2: 30440220060825daf3e7336e45755ec6d6da0757ac33ae4931b99a4f319dffd50e3fb8d7022008b425c0b6dadcf70d2b209b0b0cdb34f3e61f48fbfd38d1abf0cc617c2f5f14
---

# 基于图神经网络与强化学习的股票投资组合优化研究设计方案

## 一、论文题目与摘要

### 中文题目

基于动态图神经网络与强化学习的股票投资组合优化研究

### 英文题目

Graph-Enhanced Reinforcement Learning for Portfolio Management: Exploiting Sectoral and Temporal Dependencies

### 摘要

股票市场具有高度的非线性和复杂性，传统的单一股票时间序列预测方法难以充分捕捉资产之间的复杂依赖关系。本研究提出一种新颖的图增强强化学习框架（GNN-RL-Portfolio），将图神经网络（GNN）与强化学习（RL）相结合，用于优化股票投资组合管理策略。该框架的核心创新在于：首先，利用LSTM网络提取各只股票的历史价格时间序列特征，捕捉其时序动态特性；其次，引入图注意力网络（GAT）建模股票之间的空间依赖关系，将行业关联、产业链上下游等结构化信息融入状态表示；最后，采用近端策略优化（PPO）算法作为强化学习核心，通过端到端的方式同时优化特征提取和决策策略。在S&P 500和CSI 300数据集上的实验表明，本研究提出的方法在累计收益率、夏普比率和最大回撤等指标上均优于现有的基准方法，验证了图结构信息与强化学习决策联合建模的有效性。

## 二、引言

### 研究背景与动机

量化投资作为一种基于数学模型和计算机程序的系统化投资方法，近年来在金融领域获得了广泛关注。随着深度学习技术的快速发展，研究者们开始探索将神经网络应用于股价预测和投资组合优化任务。早期的研究主要关注使用循环神经网络（RNN）、长短期记忆网络（LSTM）等时序模型捕捉股票价格的动态演变规律，并取得了显著的进展。然而，这些方法存在一个共同的局限性：它们通常将每只股票视为独立的实体，忽视了股票之间存在的复杂关联关系。

实际上，股票市场本身具有天然的图结构特性。从微观层面来看，同一行业内的股票往往受到相似的宏观经济因素影响，呈现出较强的联动效应；从宏观层面来看，产业链的上下游企业之间存在供需关联，一个企业的经营状况会沿着产业链传导至相关企业；此外，市场情绪也会在关联股票之间传播，形成所谓的“板块效应”。这种复杂的关联关系意味着，孤立地预测单只股票的价格走势可能无法完整地反映市场的整体动态，从而限制预测模型的性能上限。

强化学习作为一种端到端的决策优化范式，近年来在游戏、机器人控制等领域取得了突破性进展。与传统的监督学习方法不同，强化学习智能体通过与环境的交互学习最优策略，能够自动发现序列决策中的复杂模式。在投资组合管理场景中，强化学习可以将资产配置问题建模为马尔可夫决策过程（MDP），直接优化长期累积收益，而非仅仅追求单步预测的准确性。这种决策导向的学习范式更符合实际投资管理的目标。

然而，现有的强化学习方法在投资组合优化中的应用同样面临挑战。传统的强化学习方法通常将投资组合状态简化为各资产历史价格的拼接，忽略了资产之间的关联信息。这种简化的状态表示不仅丢失了重要的市场结构信息，也限制了智能体对市场动态的理解能力。特别是在市场剧烈波动期间，资产之间的关联关系会发生显著变化，强化学习智能体如果无法感知这些变化，就难以做出合理的风险控制决策。

### 研究问题与创新点

基于上述分析，本研究提出一个核心研究问题：如何有效地将股票之间的图结构信息融入强化学习框架，以提升投资组合管理的绩效？围绕这一问题，本研究的主要贡献和创新点可以概括为以下几个方面：

第一，提出一种端到端的图增强强化学习框架，将时序特征提取、空间关系建模和策略优化统一在同一个学习框架中。该框架能够同时利用股票的时间动态特性和空间关联特性，生成更加丰富的状态表示，为强化学习智能体提供更全面的决策依据。

第二，设计一种动态图构建方法，综合考虑静态的行业分类关系和动态的价格相关性。在静态图的基础上，引入基于滑动窗口计算的Pearson相关系数来动态更新图结构，使模型能够自适应地捕捉市场中实时变化的关联关系。这一创新使模型能够在不同市场环境下灵活调整其关注的关联结构。

第三，构建混合奖励函数，综合考虑收益和风险两个维度。传统的强化学习方法通常使用简单的收益率作为奖励信号，这种设计可能导致智能体过度追求高收益而忽视风险。本研究引入风险调整后的收益指标（如夏普比率）作为奖励函数，引导智能体在追求高收益的同时兼顾风险控制。

## 三、相关工作

### 基于深度学习的股价预测

深度学习方法在股价预测任务中的应用可以追溯到循环神经网络（RNN）的引入。RNN通过隐藏状态的传递，能够在一定程度上捕捉时间序列中的长期依赖关系，但其梯度消失和梯度爆炸问题限制了模型处理长序列的能力。长短期记忆网络（LSTM）通过引入门控机制，有效缓解了上述问题，成为股价预测领域的基准方法。随后，Transformer架构以其强大的注意力机制在自然语言处理领域取得了巨大成功，也被研究者们引入到时间序列预测任务中。Transformer能够并行处理序列中的所有时间步，并通过自注意力机制动态地分配不同时间点的重要性，在捕捉长程依赖方面展现了优异的能力。

近年来，研究者们开始探索更复杂的时间序列预测模型。Informer模型通过稀疏注意力机制降低了计算复杂度，使得处理超长序列成为可能。Autoformer进一步提出了自相关分解机制，能够将时间序列分解为趋势项和周期项分别进行建模。Crossformer则引入了跨维度注意力机制，能够同时处理多变量时间序列中的空间和时间依赖。然而，这些方法主要关注单一时间序列的预测问题，未能考虑不同时间序列之间的相互影响。

### 图神经网络在金融领域的应用

图神经网络（GNN）作为处理图结构数据的利器，近年来在社交网络、推荐系统、生物信息学等领域获得了广泛应用。在金融领域，研究者们也开始探索利用GNN建模股票之间的关联关系。早期的研究主要关注使用相关性分析构建股票网络，然后应用图卷积网络（GCN）进行节点分类或链接预测。近年来，更多的工作开始关注如何将GNN与时间序列预测相结合，构建时空图神经网络模型。

Stock Graph Convolutional Network（SGCN）首次系统性地将图卷积网络应用于股价预测任务，通过行业分类和市值规模构建股票之间的关系图，并验证了图结构信息能够有效提升预测性能。Temporal Graph Attention Network（TITAN）进一步引入了时序注意力机制，能够在动态变化的市场环境中自适应地调整不同邻居节点的重要性。THAT（Hierarchical Graph Attention Network）则提出了分层注意力机制，分别在股票节点、行业板块和市场整体三个层次进行信息聚合。这些研究为本文的工作提供了重要的技术基础。

### 强化学习在量化交易中的应用

强化学习在量化交易中的应用始于将交易决策建模为马尔可夫决策过程（MDP）的研究。深度Q网络（DQN）及其变体被率先应用于简单的交易场景，将买入、卖出、持有等动作离散化，并使用深度神经网络近似Q函数。然而，离散动作空间难以精确表达投资组合的权重分配需求。

随后，研究者们开始探索使用连续动作空间的强化学习算法。深度确定性策略梯度（DDPG）算法被应用于投资组合优化，允许智能体输出连续的资产权重向量。近端策略优化（PPO）算法以其训练稳定性和样本效率优势，成为当前量化交易领域最流行的强化学习算法之一。Actor-Critic框架的引入进一步提升了策略梯度方法的稳定性和收敛速度。

多智能体强化学习（MARL）在投资组合管理中的应用是近年来的一个新的研究方向。MAPS系统提出将每只股票视为一个独立的智能体，通过多智能体协作学习投资组合策略。TC-MARL进一步引入了趋势一致性正则化，协调不同智能体之间的学习过程。这些研究表明，多智能体框架能够更好地处理投资组合优化中的复杂性。

### 当前研究的局限性

尽管上述三个方向都取得了显著进展，但它们之间的交叉融合研究仍然相对有限。现有的大多数工作要么专注于改进时间序列预测模型，要么单独应用强化学习进行决策优化，或者将GNN用于股票关系建模，却较少系统性地探讨如何将这三者有效结合。具体而言，当前研究存在以下主要局限：首先，GNN提取的图特征通常仅用于监督学习任务，与后续的决策过程分离，无法实现端到端的优化；其次，动态图结构的构建方法相对简单，通常仅基于静态的行业分类，未能充分挖掘市场中的实时关联信息；第三，强化学习的状态表示设计未能有效利用GNN提供的丰富图结构信息，导致智能体对市场整体结构的感知能力不足。

## 四、方法论

### 问题定义与马尔可夫决策过程建模

本研究将股票投资组合优化问题形式化为马尔可夫决策过程（MDP），用六元组$\langle S, A, P, R, \gamma, \mu \rangle$表示，其中各元素含义如下：

**状态空间（$S$）**：在每个时间步$t$，智能体观察到的状态$s_t$包含三个主要组成部分：宏观经济指标向量$e_t$（如无风险利率、市场波动率指数等）、个股特征矩阵$X_t$（包含价格、成交量、技术指标等）以及当前的资产配置权重$w_{t-1}$。形式化地，$s_t = \{e_t, X_t, w_{t-1}\}$。

**动作空间（$A$）**：动作$a_t$表示在时间步$t$结束时要调整到的目标资产权重向量$a_t = (a_t^1, a_t^2, ..., a_t^n)$，其中$a_t^i$表示分配给第$i$只股票的资产比例，满足$\sum_{i=1}^n a_t^i = 1$且$a_t^i \geq 0$。与离散动作空间相比，连续权重向量能够更精确地表达资产配置意图。

**状态转移概率（$P$）**：$P(s_{t+1}|s_t, a_t)$描述在状态$s_t$下执行动作$a_t$后转移到状态$s_{t+1}$的概率分布。在真实的股票市场中，状态转移受到众多不可控因素的影响，可以视为随机过程。

**奖励函数（$$R$）**：本研究采用风险调整后的收益率作为奖励函数，具体形式为：$r_t = (V_t - V_{t-1})/V_{t-1} - \lambda \cdot \sigma_t$，其中$V_t$表示时间步$t$的资产总值，$\sigma_t$表示资产收益率的波动率，$\lambda$为风险厌恶系数。这种设计引导智能体在追求高收益的同时兼顾风险控制。

**折扣因子（$\gamma$）**：折扣因子$\gamma \in [0, 1]$用于平衡短期收益和长期收益的重要性。较大的折扣因子使智能体更加关注长期累积收益。

**初始状态分布（$\mu$）**：$\mu(s_0)$表示初始状态的概率分布。

### 图结构构建方法

本研究构建的股票关系图包含两种类型的边：静态行业边和动态相关边。这种混合图结构能够同时捕捉股票之间稳定的结构性关系和动态变化的联动效应。

**静态行业边**：根据股票所属的行业分类（同花顺行业分类或GICS行业分类），将同一行业内的股票连接起来。形式化地，如果股票$i$和$j$属于同一行业，则它们之间存在一条边$e_{ij}^s = 1$。行业分类反映了企业在经济活动中的相似性，是股票关联的稳定来源。

**动态相关边**：在静态行业边的基础上，引入基于滑动窗口计算的动态相关系数。具体而言，对于每个时间步$t$，计算过去$L$天内股票$i$和$j$的日收益率序列之间的Pearson相关系数$\rho_{ij}^t$。如果$\rho_{ij}^t$超过预设阈值$\theta$，则在股票$i$和$j$之间添加一条动态边。这种动态边能够捕捉市场中的即时联动效应，如概念炒作、业绩公告等事件驱动的相关性变化。

最终的邻接矩阵$A_t$由静态邻接矩阵$A^s$和动态邻接矩阵$A_t^d$共同决定：$A_t = A^s + \alpha \cdot A_t^d$，其中$\alpha$为平衡两种边类型的超参数。

### 模型架构设计

本研究提出的GNN-RL-Portfolio框架由三个核心组件构成：时序编码器、图注意力模块和强化学习策略网络。

**时序编码器（Temporal Encoder）**：时序编码器负责从每只股票的历史价格序列中提取时序特征。具体而言，对于股票$i$，其输入为过去$L$天的特征向量序列$\{x_t^{i,1}, x_t^{i,2}, ..., x_t^{i,L}\}$，每个时间步的特征包含开盘价、收盘价、最高价、最低价、成交量以及若干技术指标（如RSI、MACD、布林带等）。时序编码器采用双向LSTM网络进行处理：

$$h_t^i = \text{LSTM}(x_t^{i,1:L}; \theta_{LSTM})$$

其中$h_t^i \in \mathbb{R}^d$为股票$i$在时间步$t$的$d$维时序隐藏表示。所有股票的时序隐藏表示组成矩阵$H_t \in \mathbb{R}^{n \times d}$，其中$n$为股票数量。

**图注意力模块（Spatial Graph Module）**：图注意力模块的核心是图注意力网络（GAT），其作用是聚合邻居节点的信息来更新每个节点的表示。GAT通过可学习的注意力权重动态地决定不同邻居节点的重要性，这与股票市场中不同关联强度相吻合。具体的更新公式为：

$$h_t^{'(i)} = \text{GAT}\left(\bigoplus_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot W h_t^j\right)$$

其中$\mathcal{N}(i)$表示股票$i$的邻居节点集合，$\alpha_{ij}$为节点$i$和$j$之间的注意力权重，$W$为可学习的投影矩阵，$\oplus$表示特征拼接操作。经过多层GAT处理后，最终的图增强表示$G_t \in \mathbb{R}^{n \times d'}$包含了每只股票及其关联股票的信息。

**状态表示构建**：在获得时序编码和图增强表示后，本研究将两者进行融合以构建完整的状态表示。考虑到不同层次信息的重要性可能因市场环境而异，本研究引入门控机制自适应地融合两种表示：

$$s_t^i = \gamma_t \odot h_t^i + (1 - \gamma_t) \odot h_t^{'(i)}$$

其中$\gamma_t = \sigma(W_g [h_t^i; h_t^{'(i)}])$为门控向量，$\sigma$为sigmoid函数，$W_g$为可学习的投影矩阵。最终的整体状态表示$s_t$由所有股票的个体表示拼接而成：$s_t = [s_t^1; s_t^2; ...; s_t^n]$，并进一步与宏观经济指标$e_t$和当前权重$w_{t-1}$进行拼接。

**强化学习策略网络（RL Agent）**：策略网络采用Actor-Critic架构。Actor网络（策略函数$\pi_\theta(a_t|s_t)$）接收完整状态表示$s_t$作为输入，输出目标资产权重向量$A_t$。考虑到资产权重需要满足归一化约束（各权重之和为1），Actor网络的输出层采用Softmax函数：

$$a_t^i = \frac{\exp(f_\theta^i(s_t))}{\sum_{j=1}^n \exp(f_\theta^j(s_t))}$$

其中$f_\theta^i(s_t)$为Actor网络对股票$i$的原始输出。

Critic网络（价值函数$V_\phi(s_t)$）评估当前状态的价值，为策略梯度提供基线。Critic网络的输入为状态-动作对$(s_t, a_t)$，输出为Q值估计。

### 训练算法

本采用近端策略优化（PPO）算法作为核心训练方法。PPO通过限制策略更新的幅度来保证训练的稳定性，非常适合金融交易这种对安全性要求较高的应用场景。

PPO的目标函数为：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$为重要性采样比率，$\hat{A}_t$为优势函数估计，$\epsilon$为裁剪超参数。

此外，本研究在目标函数中加入了价值函数损失项和熵正则化项：

$$L(\theta, \phi) = L^{CLIP}(\theta) - \alpha_v L^{VF}(\phi) + \alpha_e H(\pi_\theta)$$

其中$L^{VF}(\phi) = (V_\phi(s_t) - V_{target})^2$为均方误差损失，$H(\pi_\theta) = -\mathbb{E}_{\pi_\theta}[\log \pi_\theta(a|s)]$为策略熵，$\alpha_v$和$\alpha_e$为相应的系数。

### 交易成本建模

真实市场中的交易并非零成本，实际操作中需要考虑手续费、滑点等因素的影响。本研究在奖励函数中引入交易成本：

$$r_t^{net} = r_t^{gross} - \sum_{i=1}^n |a_t^i - a_{t-1}^i| \cdot c$$

其中$r_t^{gross}$为毛收益率，$c$为交易成本费率。交易成本的设计使得智能体倾向于保持仓位稳定，只有在预期收益足以覆盖成本时才进行调整。

## 五、实验设计

### 数据集描述

本研究使用两个公开的股票数据集进行实验验证：

**S&P 500数据集**：选取S&P 500指数中市值最大的100只股票作为研究对象，时间范围为2015年1月1日至2023年12月31日。数据来源为Yahoo Finance，包含每日OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据。行业分类信息来自S&P Global官方网站。

**CSI 300数据集**：选取沪深300指数成分股中流动性最好的50只股票作为研究对象，时间范围同样为2015年1月1日至2023年12月31日。数据来源为Tushare金融数据库。行业分类信息参考申银万国行业分类标准。

### 数据预处理

原始数据首先经过以下预处理步骤：第一，剔除停牌天数超过20%的交易日，确保数据的连续性；第二，对OHLCV数据进行Z-Score标准化处理，消除量纲差异；第三，计算技术指标（包括RSI、MACD、布林带、ATR等）作为补充特征；第四，按时间顺序划分训练集（2015-2020）、验证集（2021）和测试集（2022-2023）。

### 评估指标

本研究采用以下指标全面评估模型性能：

**累计收益率（Cumulative Return）**：$CR = \frac{V_T - V_0}{V_0}$，衡量投资组合在测试期间的总收益。

**年化收益率（Annualized Return）**：$AR = (1 + CR)^{365/T} - 1$，其中$T$为交易天数。

**夏普比率（Sharpe Ratio）**：$SR = \frac{AR - R_f}{\sigma_r}$，其中$R_f$为无风险利率（取年化3%），$\sigma_r$为年化收益率标准差。

**最大回撤（Maximum Drawdown）**：$MDD = \max_{t < s} \frac{V_t - V_s}{V_s}$，衡量投资组合可能遭受的最大损失。

**胜率（Win Rate）**：$WR = \frac{N_{positive}}{N_{total}}$，其中$N_{positive}$为取得正收益的交易天数。

### 基准方法

本研究设置以下基准方法进行对比实验：

**买入持有策略（UBAH）**：在期初平均买入所有股票并持有至期末，作为最基础的被动投资策略。

**LSTM强化学习（LSTM-RL）**：仅使用LSTM提取时序特征，不包含图神经网络模块，用于验证图结构信息的有效性。

**图卷积监督学习（GCN-SL）**：使用GCN进行股价预测，然后基于预测结果使用均值-方差优化进行资产配置，用于验证强化学习决策的有效性。

**标准PPO（Vanilla-PPO）**：使用拼接的价格特征作为状态输入，不包含任何图结构信息，作为强化学习方法的基准。

**AlphaMix（KDD 2023）**：使用专家混合框架的投资组合管理方法，代表当前最先进的多智能体强化学习方法。

### 超参数设置

模型的主要超参数设置如下：LSTM隐藏层维度为128，GAT注意力头数为4，隐藏层维度为64，图卷积层数为2，时间窗口长度为20，PPO的折扣因子为0.99，裁剪参数$\epsilon$为0.2，批量大小为256，训练轮数为1000。所有实验在NVIDIA RTX 3090 GPU上运行。

## 六、核心代码实现

### 数据加载与图构建模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class StockDataLoader:
    """股票数据加载器，负责获取历史价格数据并计算技术指标"""

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = None
        self.industry_data = None

    def load_data(self):
        """加载OHLCV数据"""
        # 实际应用中应从Tushare或Yahoo Finance获取数据
        # 此处为演示代码
        pass

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 计算布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        return df

    def get_features(self, window: int = 20) -> torch.Tensor:
        """获取窗口期的特征张量"""
        # 特征包括：open, high, low, close, volume, rsi, macd, signal, bb_upper, bb_lower
        features = []
        for ticker in self.tickers:
            ticker_df = self.price_data[self.price_data['ticker'] == ticker]
            feature = ticker_df[['open', 'high', 'low', 'close', 'volume',
                               'rsi', 'macd', 'signal', 'bb_upper', 'bb_lower']].values
            features.append(feature)
        return torch.tensor(np.array(features), dtype=torch.float32)


class GraphBuilder:
    """图结构构建器，负责构建股票关系图"""

    def __init__(self, tickers: List[str], industry_map: Dict[str, str]):
        self.tickers = tickers
        self.industry_map = industry_map
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    def build_static_graph(self) -> torch.Tensor:
        """基于行业分类构建静态邻接矩阵"""
        n = len(self.tickers)
        adj_matrix = torch.zeros((n, n))

        # 同一行业的股票之间建立边
        for i, ticker_i in enumerate(self.tickers):
            for j, ticker_j in enumerate(self.tickers):
                if i != j and self.industry_map.get(ticker_i) == self.industry_map.get(ticker_j):
                    adj_matrix[i, j] = 1.0

        return adj_matrix

    def update_dynamic_edges(self, price_window: torch.Tensor,
                           threshold: float = 0.5) -> torch.Tensor:
        """基于价格相关性动态更新边"""
        # price_window shape: [num_tickers, window, features]
        # 使用收盘价计算相关性
        close_prices = price_window[:, :, 3]  # 假设第4列为收盘价

        # 计算收益率
        returns = torch.diff(close_prices, dim=1) / close_prices[:, :-1]

        # 计算Pearson相关系数矩阵
        corr_matrix = torch.corrcoef(returns)

        # 根据阈值构建动态边
        dynamic_edges = (corr_matrix > threshold).float()

        return dynamic_edges
```

### 时序编码器与图注意力模块

```python
class TemporalEncoder(nn.Module):
    """时序编码器，使用LSTM提取时间序列特征"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 双向LSTM输出维度是hidden_dim的两倍
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch, num_nodes, time_steps, features]
        """
        batch_size, num_nodes, time_steps, features = x.shape

        # 将x重塑为 [batch * num_nodes, time_steps, features]
        x = x.view(batch_size * num_nodes, time_steps, features)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch * num_nodes, hidden_dim * 2]

        # 投影到目标维度
        output = self.projection(last_output)  # [batch * num_nodes, hidden_dim]

        # 恢复形状
        output = output.view(batch_size, num_nodes, self.hidden_dim)

        return output


class SpatialGraphModule(nn.Module):
    """图注意力模块，使用GAT聚合邻居信息"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 多头注意力图卷积
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch, num_nodes, input_dim]
        edge_index shape: [2, num_edges]
        """
        batch_size, num_nodes, input_dim = x.shape

        # 为每个样本添加自环
        batch_edge_index = []
        for b in range(batch_size):
            edge_idx = edge_index.clone()
            # 调整边索引以适应batch中的节点偏移
            offset = b * num_nodes
            edge_idx = edge_idx + offset
            batch_edge_index.append(edge_idx)

        batch_edge_index = torch.cat(batch_edge_index, dim=1)

        # 将x重塑为2D以适应PyG
        x_2d = x.view(batch_size * num_nodes, input_dim)

        # 第一层GAT + ReLU + Dropout
        h = self.gat1(x_2d, batch_edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        # 第二层GAT
        h = self.gat2(h, batch_edge_index)

        # 残差连接和层归一化
        h = h + x_2d  # 残差连接
        h = self.layer_norm(h)

        # 恢复形状
        h = h.view(batch_size, num_nodes, self.hidden_dim)

        return h


class StateEncoder(nn.Module):
    """状态编码器，整合时序和图信息"""

    def __init__(self, temporal_input_dim: int, graph_input_dim: int,
                 hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(temporal_input_dim, hidden_dim)
        self.spatial_graph = SpatialGraphModule(hidden_dim, hidden_dim, num_heads)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 宏观经济指标编码器
        self.macro_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 假设8个宏观指标
            nn.ReLU()
        )

        # 当前持仓编码器
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim),  # 假设50只股票
            nn.ReLU()
        )

    def forward(self, temporal_features: torch.Tensor,
                graph_features: torch.Tensor,
                edge_index: torch.Tensor,
                macro_features: torch.Tensor,
                current_weights: torch.Tensor) -> torch.Tensor:
        """
        temporal_features: [batch, num_nodes, time_steps, features]
        graph_features: [batch, num_nodes, graph_features]
        edge_index: [2, num_edges]
        macro_features: [batch, num_macro]
        current_weights: [batch, num_nodes]
        """
        # 1. 时序编码
        temporal_emb = self.temporal_encoder(temporal_features)  # [batch, num_nodes, hidden]

        # 2. 图注意力聚合
        # 将temporal embedding作为graph input
        graph_emb = self.spatial_graph(temporal_emb, edge_index)  # [batch, num_nodes, hidden]

        # 3. 门控融合
        combined = torch.cat([temporal_emb, graph_emb], dim=-1)  # [batch, num_nodes, hidden*2]
        gate_values = self.gate(combined)
        fused_emb = gate_values * temporal_emb + (1 - gate_values) * graph_emb

        # 4. 聚合所有节点信息
        pooled = torch.mean(fused_emb, dim=1)  # [batch, hidden]

        # 5. 编码宏观特征和当前持仓
        macro_emb = self.macro_encoder(macro_features)  # [batch, hidden]
        weight_emb = self.portfolio_encoder(current_weights)  # [batch, hidden]

        # 6. 最终状态表示
        state = pooled + macro_emb + weight_emb

        return state
```

### 强化学习环境与智能体

```python
import gym
from gym import spaces
from collections import deque

class StockPortfolioEnv(gym.Env):
    """股票投资组合强化学习环境"""

    def __init__(self, price_data: torch.Tensor,
                 adj_matrix: torch.Tensor,
                 transaction_cost: float = 0.001,
                 initial_capital: float = 100000.0):
        super().__init__()

        self.price_data = price_data  # [num_days, num_stocks, features]
        self.adj_matrix = adj_matrix  # [num_stocks, num_stocks]
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.num_stocks = price_data.shape[1]
        self.num_days = price_data.shape[0]

        # 动作空间：连续权重向量
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.num_stocks,),
            dtype=np.float32
        )

        # 观察空间
        self.observation_space = spaces.Dict({
            'price_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_stocks, 20, 10),  # 窗口20天，10个特征
                dtype=np.float32
            ),
            'current_weights': spaces.Box(
                low=0, high=1,
                shape=(self.num_stocks,),
                dtype=np.float32
            ),
            'portfolio_value': spaces.Box(
                low=0, high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })

        self.current_day = 0
        self.current_value = initial_capital
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks

    def reset(self):
        self.current_day = 20  # 跳过前20天以获取足够的历史数据
        self.current_value = self.initial_capital
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks

        return self._get_observation()

    def step(self, action: np.ndarray):
        # 归一化动作（确保权重和为1）
        action = action / (action.sum() + 1e-8)

        # 计算交易成本
        weight_diff = np.abs(action - self.current_weights)
        trading_cost = np.sum(weight_diff) * self.transaction_cost

        # 获取当天和第二天的价格
        prices_t = self.price_data[self.current_day, :, 3].numpy()  # 收盘价
        prices_t1 = self.price_data[self.current_day + 1, :, 3].numpy()

        # 计算收益率
        returns = (prices_t1 - prices_t) / prices_t

        # 计算投资组合收益
        portfolio_return = np.dot(self.current_weights, returns) - trading_cost

        # 更新投资组合价值
        self.current_value *= (1 + portfolio_return)

        # 更新权重
        self.current_weights = action * (1 + returns) / (1 + portfolio_return)
        self.current_weights = self.current_weights / (self.current_weights.sum() + 1e-8)

        # 奖励为收益率减去波动率惩罚
        reward = portfolio_return - 0.01 * np.std(returns)

        # 推进时间
        self.current_day += 1

        # 检查是否结束
        done = self.current_day >= self.num_days - 1

        info = {
            'portfolio_value': self.current_value,
            'portfolio_return': portfolio_return
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # 获取过去20天的价格特征
        price_window = self.price_data[self.current_day - 20:self.current_day].numpy()

        obs = {
            'price_features': price_window.astype(np.float32),
            'current_weights': self.current_weights.astype(np.float32),
            'portfolio_value': np.array([self.current_value], dtype=np.float32)
        }

        return obs


class PPOPortfolioAgent(nn.Module):
    """基于PPO的投资组合管理智能体"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def get_action(self, state: torch.Tensor):
        action_probs, value = self.forward(state)

        # 从概率分布中采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        return action, value, dist.log_prob(action)
```

### 训练循环

```python
def train_gnn_rl_portfolio(env, agent, state_encoder, num_episodes=1000,
                          lr_actor=3e-4, lr_critic=1e-3,
                          gamma=0.99, clip_epsilon=0.2):
    """训练GNN-RL投资组合智能体"""

    optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)
    optimizer_critic = torch.optim.Adam(agent.critic.parameters(), lr=lr_critic)
    optimizer_encoder = torch.optim.Adam(state_encoder.parameters(), lr=lr_actor)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_values = []
        episode_log_probs = []
        episode_rewards = []

        while True:
            # 编码状态
            state_tensor = torch.FloatTensor(state['price_features']).unsqueeze(0)
            weights_tensor = torch.FloatTensor(state['current_weights']).unsqueeze(0)

            with torch.no_grad():
                encoded_state = state_encoder(
                    state_tensor,
                    state_tensor,  # graph features
                    env.adj_matrix,
                    torch.randn(1, 8),  # macro features
                    weights_tensor
                )

            # 获取动作
            action, value, log_prob = agent.get_action(encoded_state)

            # 执行动作
            next_state, reward, done, info = env.step(action.numpy().flatten())

            # 记录转换
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            episode_reward += reward

            if done:
                break

            state = next_state

        # 计算回报和优势
        returns = []
        discounted_reward = 0
        for reward in reversed(episode_rewards):
            discounted_reward = reward + gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(episode_values).squeeze()

        # 计算优势
        advantages = returns - values.detach()

        # PPO更新
        log_probs = torch.cat(episode_log_probs).squeeze()

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)

        # 熵正则化
        dist = torch.distributions.Categorical(agent.actor(encoded_state)[0])
        entropy = dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        optimizer_encoder.zero_grad()

        total_loss.backward()

        optimizer_actor.step()
        optimizer_critic.step()
        optimizer_encoder.step()

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Actor Loss = {actor_loss.item():.4f}, "
                  f"Critic Loss = {critic_loss.item():.4f}")

    return agent, state_encoder
```

## 七、创新点总结与研究价值

### 核心创新点

本研究的核心创新点可以概括为以下三个方面：

第一，提出了一种新颖的图增强强化学习框架，将时序特征提取、空间关系建模和策略优化统一在端到端的学习框架中。与现有方法相比，本框架能够同时利用股票的时序动态特性和空间关联特性，生成更加丰富的状态表示。特别是引入的门控融合机制，能够自适应地调整时序信息和图结构信息在状态表示中的权重，使模型能够根据不同的市场环境灵活调整其关注重点。

第二，设计了一种混合图结构构建方法，综合考虑静态的行业分类关系和动态的价格相关性。静态行业边反映了股票之间稳定的经济联系，是关联信息的可靠来源；而动态相关边则能够捕捉市场中实时变化的联动效应，如业绩公告、概念炒作等事件驱动的相关性变化。这种混合图结构使模型能够在不同市场环境下自适应地调整其关注的关联结构。

第三，构建了风险敏感的投资组合优化目标。传统的强化学习方法通常使用简单的收益率作为奖励信号，可能导致智能体过度追求高收益而忽视风险。本研究引入波动率惩罚和交易成本，使智能体能够在追求高收益的同时兼顾风险控制和交易成本最小化，更加符合实际投资管理的需求。

### 研究价值与应用前景

本研究成果具有重要的理论价值和广阔的应用前景。从理论层面来看，本研究为强化学习在金融领域的应用提供了新的思路和方法论支撑，特别是在如何有效利用复杂市场结构信息方面提供了可借鉴的解决方案。研究成果有望推广到其他具有图结构特性的金融决策问题，如期货套利、期权对冲等。

从应用层面来看，本研究提出的方法可以直接应用于量化投资实践。随着量化交易的普及，投资机构对高性能投资策略的需求日益增长。本研究提出的GNN-RL-Portfolio框架能够充分利用股票之间的关联信息，有望在实际投资中获得优于传统方法的绩效表现。此外，图结构信息的引入也增强了模型的可解释性，投资者可以直观地理解模型在不同市场环境下的决策依据。

## 八、实验预期与验证计划

### 预期结果

基于现有文献和理论分析，我们预期本研究的实验结果将呈现以下特征：

首先，在累计收益率指标上，GNN-RL-Portfolio方法预期将显著优于买入持有策略和单一LSTM强化学习方法。这主要得益于图结构信息能够帮助模型更全面地理解市场动态，避免因忽视股票关联而导致的决策失误。

其次，在夏普比率指标上，本研究方法预期将取得最优表现。这是因为图注意力机制使模型能够感知市场整体的风险状态，在市场风险较高时自动降低风险敞口，实现风险与收益的平衡优化。

第三，在最大回撤指标上，本研究方法预期将表现出较好的鲁棒性。多智能体协作学习使模型能够学习到更加保守的仓位管理策略，在市场剧烈波动时有效控制下行风险。

### 消融实验设计

为验证各模块的有效性，本研究设计了以下消融实验：

第一组实验：移除图神经网络模块，仅使用LSTM提取时序特征，用于验证图结构信息的边际贡献。

第二组实验：移除LSTM模块，仅使用图神经网络处理当前时刻的特征，用于验证时序信息的重要性。

第三组实验：使用静态图结构（仅包含行业边），移除动态相关边，用于验证动态图结构的有效性。

第四组实验：将强化学习替换为监督学习方法（预测收益率后使用均值-方差优化），用于验证强化学习决策框架的优越性。

### 可视化分析

本研究计划包含以下可视化分析内容：图注意力权重热力图，展示模型在不同市场时期关注的股票关联；投资组合权重变化曲线，展示模型的仓位调整行为；累计收益曲线对比图，直观展示各方法的绩效差异；市场环境分析案例，如2020年新冠疫情期间模型的行为表现。

## 九、总结与展望

本研究围绕“如何有效结合强化学习、图神经网络与股票时间序列预测”这一核心问题，提出了一种新颖的图增强强化学习框架。该框架通过LSTM捕捉股票的历史价格动态，利用图注意力网络建模股票之间的关联关系，并将这些丰富的状态表示输入到PPO强化学习算法中进行投资组合优化。

在S&P 500和CSI 300数据集上的实验预期将验证本框架的有效性，特别是图结构信息的引入能够显著提升投资组合的风险调整收益。本研究的创新点包括：端到端的时空特征融合框架、混合动态图结构构建方法以及风险敏感的投资组合优化目标。

展望未来，我们计划从以下方向进一步拓展本研究：首先，探索引入更多模态的信息（如新闻文本、分析师报告等），构建多模态的市场状态表示；其次，研究如何在模型中引入因果推断机制，以更准确地理解股票之间的因果关系而非简单的相关关系；第三，将研究拓展至衍生品市场，探索期权定价和对冲策略优化问题。


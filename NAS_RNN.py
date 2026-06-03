# ====== 导入依赖库 ======
import torch  # PyTorch 主库，用于张量运算与自动微分
import torch.nn as nn  # 神经网络模块，提供各类网络层
import torch.optim as optim  # 优化器模块（Adam、SGD等）
import numpy as np  # 数值计算库
import random  # Python 标准库，用于随机采样
from sklearn.datasets import make_regression  # 生成回归数据集
from sklearn.model_selection import train_test_split  # 划分训练/验证集
from sklearn.preprocessing import StandardScaler  # 数据标准化

# 固定随机种子，保证搜索过程可复现
torch.manual_seed(42)  # PyTorch 随机数种子
np.random.seed(42)     # NumPy 随机数种子
random.seed(42)        # Python random 模块种子

# ====== 搜索空间定义 ======
# 神经架构搜索的离散超参数候选集合，控制器将从中组合选择最优架构
search_space = {
    'num_hidden_layers': [1, 2, 3],              # 隐藏层数量候选
    'hidden_layer_size': [32, 64, 128],          # 隐藏层神经元数候选
    'activation_function': ['ReLU', 'Tanh', 'LeakyReLU'],  # 激活函数候选
    'learning_rate': [0.01, 0.001, 0.0001],     # 学习率候选
    'optimizer': ['Adam', 'SGD'],                # 优化器候选
    'dropout_rate': [0.0, 0.2, 0.5]             # Dropout 失活率候选
}

# 字符串到激活函数类的映射，便于动态构造网络层
activation_map = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU
}

# 字符串到优化器类的映射，根据架构选择对应优化器
optimizer_map = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
}

# ====== 构建候选模型 ======
def build_model(architecture, input_dim=10, output_dim=1):
    """
    根据架构字典动态构建神经网络模型。
    architecture: 包含超参数的字典（层数、隐藏维度、激活函数、Dropout等）
    input_dim: 输入特征维度
    output_dim: 输出维度（回归任务通常为1）
    """
    layers = []
    in_features = input_dim

    # 按照 num_hidden_layers 指定的层数逐层堆叠：Linear -> 激活 -> (可选)Dropout
    for _ in range(architecture['num_hidden_layers']):
        layers.append(nn.Linear(in_features, architecture['hidden_layer_size']))
        layers.append(activation_map[architecture['activation_function']]())
        if architecture['dropout_rate'] > 0:
            layers.append(nn.Dropout(architecture['dropout_rate']))
        in_features = architecture['hidden_layer_size']

    layers.append(nn.Linear(in_features, output_dim))  # 输出层
    return nn.Sequential(*layers)

# ====== 评估架构性能 ======
def evaluate_architecture(architecture, X_train, y_train, X_val, y_val, num_epochs=20):
    """
    训练给定架构的模型，返回验证集上的MSE损失。
    损失越小，架构越好。
    """
    try:
        model = build_model(architecture, X_train.shape[1], y_train.shape[1])
        criterion = nn.MSELoss()
        optimizer_class = optimizer_map[architecture['optimizer']]
        optimizer = optimizer_class(model.parameters(), lr=architecture['learning_rate'])

        # 训练阶段：在训练集上迭代优化
        model.train()
        for _ in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # 验证阶段：在验证集上评估模型性能
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        return val_loss.item()
    except Exception as e:
        # 防御式处理：若架构构建/训练失败，返回极差分数，让搜索自然淘汰该架构
        print(f"评估架构时出错: {e}")
        return float('inf')

# ====== RNN控制器 ======
class ArchitectureController(nn.Module):
    """
    使用RNN作为控制器，逐步生成架构的每个超参数。
    对于搜索空间中的每个维度，控制器输出一个分类分布，
    从中采样得到该维度的具体取值。

    工作流程：
      1. 接收时间步索引（当前处理第几个超参数维度）
      2. 通过嵌入层转换为向量
      3. 通过LSTM编码，生成隐状态
      4. 通过策略头生成各维度的logits（未归一化概率）
      5. 采样得到具体的超参数取值
    """
    def __init__(self, search_space):
        super(ArchitectureController, self).__init__()
        self.search_space = search_space
        self.keys = list(search_space.keys())  # 超参数维度名称列表
        self.vocab_size = [len(search_space[key]) for key in self.keys]  # 各维度的候选数
        self.num_actions = len(self.keys)  # 总共有多少个超参数维度

        # 时间步嵌入层：将维度索引映射到32维向量空间
        # 让RNN能够区分不同的时间步（处理不同的超参数维度）
        self.embedding = nn.Embedding(self.num_actions, 32)

        # LSTM编码器：处理时间序列信息，输出64维隐状态
        # 这样RNN可以学到"前面选择了什么超参数"对"后续选择"的影响
        self.rnn = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        # 策略头列表：为每个超参数维度创建一个线性层
        # 输入是LSTM的隐状态（64维），输出是该维度的候选数（vocab_size）
        # 输出的logits经过softmax后形成概率分布，用于采样
        self.policy_heads = nn.ModuleList([
            nn.Linear(64, vs) for vs in self.vocab_size
        ])

    def forward(self, step_indices, hidden_state=None):
        """
        step_indices: 当前时间步的索引 (batch_size, seq_len)
        hidden_state: LSTM的隐状态（包含h和c两个张量）
        返回: logits列表（每个元素对应一个超参数维度）和新的隐状态
        """
        # 嵌入时间步信息：将索引转换为32维向量
        embedded = self.embedding(step_indices)  # (batch_size, seq_len, 32)

        # 通过LSTM编码：处理序列信息，输出隐状态
        rnn_output, hidden_state = self.rnn(embedded, hidden_state)  # (batch_size, seq_len, 64)

        # 取最后一个时间步的输出作为该步的特征表示
        last_output = rnn_output[:, -1, :]  # (batch_size, 64)

        # 通过各个策略头生成logits
        # 每个策略头输出该维度的未归一化概率（logits）
        logits = [head(last_output) for head in self.policy_heads]

        return logits, hidden_state

# ====== 强化学习搜索 ======
def run_rl_search(search_space, X_train, y_train, X_val, y_val,
                  num_episodes=20, num_epochs_per_arch=15):
    """
    使用强化学习搜索最优架构。

    算法流程（REINFORCE）：
      1. 初始化RNN控制器
      2. 每个episode中：
         a) 控制器逐个生成架构的各个超参数（通过采样）
         b) 评估该架构的性能（验证损失）
         c) 计算奖励：reward = -val_loss（损失越小奖励越大）
         d) 计算策略损失：-sum(log_prob) * reward
         e) 反向传播更新控制器参数
      3. 记录最优架构

    参数：
      num_episodes: 搜索的总轮数（每轮生成并评估一个架构）
      num_epochs_per_arch: 训练每个架构的轮数
    """
    # 初始化控制器和优化器
    controller = ArchitectureController(search_space)
    controller_optimizer = optim.Adam(controller.parameters(), lr=0.01)

    best_loss = float('inf')
    best_architecture = None
    episode_rewards = []

    for episode in range(num_episodes):
        # 梯度清零（准备新的反向传播）
        controller_optimizer.zero_grad()

        # 初始化LSTM隐状态为None，LSTM会自动初始化为零
        hidden_state = None

        # 存储本episode中采样的对数概率和生成的架构
        log_probs = []
        architecture = {}

        # 逐个超参数维度生成架构
        # 这样做的好处是RNN可以学到维度间的依赖关系
        for i, key in enumerate(controller.keys):
            # 创建时间步索引张量：表示当前处理第i个维度
            step_idx = torch.tensor([[i]], dtype=torch.long)

            # 控制器生成该维度的logits（未归一化概率）
            logits, hidden_state = controller(step_idx, hidden_state)

            # 为该维度创建分类分布（Categorical distribution）
            # logits[i] 对应第i个维度的候选值的未归一化概率
            dist = torch.distributions.Categorical(logits=logits[i])

            # 从分布中采样一个动作（超参数取值的索引）
            action_idx = dist.sample()

            # 记录该采样的对数概率（用于策略梯度计算）
            log_probs.append(dist.log_prob(action_idx))

            # 存储采样的超参数值（从候选集中取出）
            architecture[key] = search_space[key][action_idx.item()]

        # 评估生成的架构：训练模型并返回验证损失
        val_loss = evaluate_architecture(architecture, X_train, y_train, X_val, y_val,
                                        num_epochs=num_epochs_per_arch)

        # 计算奖励：损失越小，奖励越大（负号反转）
        reward = -val_loss
        episode_rewards.append(reward)

        # 计算策略损失（REINFORCE算法的核心）
        # 目标：最大化 E[sum(log_prob) * reward]
        # 等价于最小化 -E[sum(log_prob) * reward]
        # 直观理解：如果奖励高（损失低），增加这些动作的概率；反之降低
        policy_loss = -torch.sum(torch.stack(log_probs)) * reward

        # 反向传播：计算梯度
        policy_loss.backward()

        # 梯度下降：更新控制器参数
        controller_optimizer.step()

        # 更新全局最优架构
        if val_loss < best_loss:
            best_loss = val_loss
            best_architecture = architecture.copy()

        # 定期输出搜索进度
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            print(f"Episode {episode+1}/{num_episodes} | 最优损失: {best_loss:.4f} | "
                  f"当前损失: {val_loss:.4f} | 平均奖励: {avg_reward:.4f}")
            print(f"  当前架构: {architecture}")

    return best_architecture, best_loss

# ====== 数据准备 ======
print("准备数据...")
# 生成合成回归数据集：300个样本，10维特征，加0.1噪声
X, y = make_regression(n_samples=300, n_features=10, noise=0.1, random_state=42)
y = y.reshape(-1, 1)  # 转成 (N, 1) 的二维矩阵

# 标准化特征和目标值（零均值、单位方差），有助于神经网络训练稳定
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 划分训练集与验证集（80% / 20%）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量（float32精度）
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ====== 运行NAS搜索 ======
print("\n开始强化学习NAS搜索...\n")
# 启动搜索：20个episode，每个架构训练15轮
best_arch, best_loss = run_rl_search(
    search_space, X_train, y_train, X_val, y_val,
    num_episodes=20, num_epochs_per_arch=15
)

# ====== 输出搜索结果 ======
print("\n" + "="*60)
print("搜索完成！")
print("="*60)
print(f"最佳架构: {best_arch}")
print(f"最佳验证损失: {best_loss:.6f}")

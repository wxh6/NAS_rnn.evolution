# ====== 导入依赖库 ======
import torch  # PyTorch 主库，用于张量运算与自动微分
import torch.nn as nn  # 神经网络模块，提供常见层与损失函数
import torch.optim as optim  # 优化器模块，例如 Adam、SGD
import random  # Python 标准库，用于随机选择架构超参数
import numpy as np  # 数值计算库，主要用于数组操作和随机种子设置
from sklearn.datasets import make_regression  # 生成回归任务的合成数据集
from sklearn.model_selection import train_test_split  # 划分训练集与验证集
from sklearn.preprocessing import StandardScaler  # 数据标准化（零均值、单位方差）

# 固定随机种子，保证每次运行得到相同的搜索结果（结果可复现）
torch.manual_seed(42)  # 固定 PyTorch 随机数（影响参数初始化等）
np.random.seed(42)     # 固定 NumPy 随机数（影响数据生成、划分）
random.seed(42)        # 固定 Python random 模块（影响架构采样）

# ====== 定义搜索空间（NAS 中候选超参数的离散取值集合）======
# 神经架构搜索（Neural Architecture Search）会在以下离散候选项中组合搜索最优架构
search_space = {
    "num_hidden_layers": [1, 2, 3],                          # 隐藏层数量候选
    "hidden_layer_size": [16, 32, 64, 128],                  # 每个隐藏层的神经元个数候选
    "activation_function": ["ReLU", "Tanh", "LeakyReLU"],    # 激活函数候选
    "learning_rate": [0.1, 0.01, 0.001],                     # 学习率候选
    "optimizer": ["Adam", "SGD"],                            # 优化器候选
    "dropout_rate": [0.0, 0.2, 0.5]                          # Dropout 失活率候选（0 表示不使用 Dropout）
}

# 字符串到激活函数类的映射，便于根据搜索结果动态构造网络层
activation_map = {
    "ReLU": nn.ReLU,            # 修正线性单元，常用且收敛快
    "Tanh": nn.Tanh,            # 双曲正切，输出范围 (-1, 1)
    "LeakyReLU": nn.LeakyReLU   # 带泄漏的 ReLU，缓解神经元死亡问题
}

# 字符串到优化器类的映射，根据架构选择对应优化器
optimizer_map = {
    "Adam": optim.Adam,  # 自适应矩估计优化器，对学习率不敏感
    "SGD": optim.SGD     # 随机梯度下降，需要更精细调参
}

# ====== 根据架构描述构建模型 ======
def build_model(arch, input_dim, output_dim):
    """
    根据架构字典 arch 动态构建一个 nn.Sequential 模型。
    arch: 单个候选架构（包含层数、隐藏维度、激活函数、Dropout 等超参数）
    input_dim:  输入特征维度
    output_dim: 输出维度（回归任务通常为 1）
    """
    layers = []                  # 存放所有顺序层
    in_features = input_dim      # 当前层的输入维度，从 input_dim 开始向后传播
    # 按照 num_hidden_layers 指定的层数逐层堆叠：Linear -> 激活 -> (可选)Dropout
    for _ in range(arch["num_hidden_layers"]):
        layers.append(nn.Linear(in_features, arch["hidden_layer_size"]))      # 全连接层
        layers.append(activation_map[arch["activation_function"]]())          # 激活函数（实例化）
        if arch["dropout_rate"] > 0:
            layers.append(nn.Dropout(arch["dropout_rate"]))                   # 可选的 Dropout 正则化
        in_features = arch["hidden_layer_size"]                               # 更新下一层的输入维度
    layers.append(nn.Linear(in_features, output_dim))   # 最后一层输出层（无激活，回归任务直接输出）
    return nn.Sequential(*layers)                       # 用 Sequential 容器把层打包成模型

# ====== 评估某个架构的验证集性能 ======
def evaluate_architecture(arch, X_train, y_train, X_val, y_val, num_epochs=30):
    """
    给定一个架构，在训练集上训练若干轮，并返回其在验证集上的 MSE 损失。
    损失越小代表架构越好（NAS 会选择损失最小者）。
    """
    try:
        # 根据搜索空间动态搭建模型与优化器
        model = build_model(arch, X_train.shape[1], y_train.shape[1])
        criterion = nn.MSELoss()                                  # 回归任务使用均方误差损失
        optimizer_class = optimizer_map[arch["optimizer"]]        # 选择优化器类
        optimizer = optimizer_class(model.parameters(), lr=arch["learning_rate"])

        # ---- 训练阶段 ----
        # 注：此处使用全批量梯度下降（一次性把全部训练数据喂入模型），数据量小可接受
        for _ in range(num_epochs):
            model.train()                  # 切换到训练模式（启用 Dropout 等）
            optimizer.zero_grad()          # 清零上一步累积的梯度
            outputs = model(X_train)       # 前向传播
            loss = criterion(outputs, y_train)  # 计算损失
            loss.backward()                # 反向传播，自动求梯度
            optimizer.step()               # 根据梯度更新参数

        # ---- 验证阶段 ----
        model.eval()                       # 切换到评估模式（关闭 Dropout）
        with torch.no_grad():              # 禁用梯度计算，节省显存并加速
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        return val_loss.item()             # 转成 Python 原生 float 返回
    except Exception as e:
        # 防御式处理：若某个架构在构建/训练阶段抛错，则返回极差分数（无穷大），
        # 让进化算法自然将其淘汰，而不是中断整个搜索流程。
        print(f"评估架构时出错: {e}")
        return float("inf")

# ====== 随机生成一个候选架构 ======
def random_architecture():
    """从搜索空间的每个维度中随机选择一个值，组合成一个完整的架构描述字典。"""
    return {k: random.choice(v) for k, v in search_space.items()}

# ====== 进化算法：基于上一代的适应度生成下一代种群 ======
def evolve(population, fitness_scores, mutation_rate=0.3):
    """
    简化版进化策略：
      1) 选择（Selection）：保留适应度排名前 50% 的个体作为“幸存者/父代”；
      2) 变异（Mutation）：从幸存者中复制并随机改动一个超参数，生成“后代”；
      3) 合并幸存者与后代，组成新一代种群（保持种群大小不变）。
    fitness_scores 与 population 一一对应，分数越小越好（验证损失）。
    mutation_rate: 复制父代后发生变异的概率。
    """
    # 按适应度（损失）升序排序，损失越低排名越靠前
    sorted_pop = [x for _, x in sorted(zip(fitness_scores, population))]
    survivors = sorted_pop[:len(population)//2]   # 取前一半作为优良个体保留

    # 通过对幸存者复制 + 概率性变异，生成补足种群规模的新个体
    offspring = []
    while len(offspring) + len(survivors) < len(population):
        parent = random.choice(survivors).copy()        # 随机挑一个幸存者作为父代（注意 .copy() 防止改写原字典）
        if random.random() < mutation_rate:             # 以 mutation_rate 概率发生变异
            key = random.choice(list(search_space.keys()))  # 随机选择一个待变异的超参数
            parent[key] = random.choice(search_space[key])  # 在该超参数候选集中随机重选一个值
        offspring.append(parent)
    return survivors + offspring  # 新一代种群 = 幸存者 + 后代

# ====== 数据准备 ======
# 使用 sklearn 生成一个合成回归数据集：500 条样本，每条 10 维特征，加 0.1 噪声
X, y = make_regression(n_samples=500, n_features=10, noise=0.1)
y = y.reshape(-1, 1)  # 把 y 从一维向量转成 (N, 1) 的二维矩阵，方便后续与模型输出对齐

# 对特征 X 与目标 y 分别做标准化（零均值、单位方差），有助于神经网络训练稳定
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 划分训练集与验证集（80% / 20%）；random_state 固定保证划分可复现
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 将 NumPy 数组转换为 PyTorch 张量（dtype=float32 与网络默认权重精度匹配）
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ====== 主搜索流程 ======
POP_SIZE = 6      # 种群规模：每一代同时评估的架构数量
GENERATIONS = 5   # 进化代数：搜索循环的总轮数

# 初始化第 0 代种群：随机采样 POP_SIZE 个架构
population = [random_architecture() for _ in range(POP_SIZE)]

# 主循环：迭代 GENERATIONS 代，每代评估当前种群并通过进化操作产生新一代
for gen in range(GENERATIONS):
    print(f"\n=== 第 {gen+1} 代 ===")
    fitness_scores = []
    # 对每个架构进行训练与验证，记录其验证损失作为适应度
    for arch in population:
        score = evaluate_architecture(arch, X_train, y_train, X_val, y_val)
        fitness_scores.append(score)
        print(f"架构: {arch}, 验证损失: {score:.4f}")

    # 基于本代适应度，进化生成下一代种群
    population = evolve(population, fitness_scores)

# ====== 输出最终最佳架构 ======
# 注：最后又对种群跑了一次评估，以选出最终代里损失最小的架构
final_scores = [evaluate_architecture(arch, X_train, y_train, X_val, y_val) for arch in population]
best_arch = population[int(np.argmin(final_scores))]  # argmin 找到损失最小的索引
print("\n最佳架构:", best_arch)

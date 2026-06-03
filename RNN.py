"""
基于强化学习（REINFORCE）的网络架构搜索（NAS）示例。

核心思路：
1. 用一个 RNN（这里用 LSTMCell）作为 Controller，逐步采样网络结构决策；
2. 每次采样得到一个架构后，调用外部评估函数得到 reward（如验证集准确率）；
3. 用 policy gradient（REINFORCE）更新 Controller，让它更倾向于高 reward 架构。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class SearchSpace:
    """定义架构搜索空间。"""

    num_layers_choices: Sequence[int] = (1, 2, 3, 4)
    hidden_size_choices: Sequence[int] = (32, 64, 128, 256)
    activation_choices: Sequence[str] = ("relu", "tanh", "gelu")


class RNNController(nn.Module):
    """
    RNN 控制器：逐步输出多个架构决策的分布。

    本示例决策顺序固定为：
    step 1 -> num_layers
    step 2 -> hidden_size
    step 3 -> activation
    """

    def __init__(self, search_space: SearchSpace, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.search_space = search_space
        self.hidden_dim = hidden_dim

        # START token，用于第一步输入
        self.start_token = nn.Parameter(torch.randn(embed_dim))

        # 统一动作嵌入层：动作 ID -> 向量
        # 这里使用一个共享 embedding，容量给大一些即可
        self.action_embed = nn.Embedding(128, embed_dim)

        self.rnn_cell = nn.LSTMCell(embed_dim, hidden_dim)

        # 三个决策头，对应三步输出 logits
        self.num_layers_head = nn.Linear(hidden_dim, len(search_space.num_layers_choices))
        self.hidden_size_head = nn.Linear(hidden_dim, len(search_space.hidden_size_choices))
        self.activation_head = nn.Linear(hidden_dim, len(search_space.activation_choices))

    def _step(
        self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor, head: nn.Linear
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步前向：更新 RNN 状态并输出当前决策 logits。"""
        h_t, c_t = self.rnn_cell(x_t, (h_t, c_t))
        logits = head(h_t)
        return logits, h_t, c_t

    @torch.no_grad()
    def sample_architecture(self) -> Dict[str, object]:
        """
        仅采样架构，不用于训练（无梯度）。
        返回可读的架构字典。
        """
        arch, _, _ = self.forward_and_sample(with_grad=False)
        return arch

    def forward_and_sample(self, with_grad: bool = True):
        """
        采样一个架构，并返回：
        - architecture: 可读架构
        - log_prob_sum: 所有决策 log_prob 之和（用于 REINFORCE）
        - entropy_sum: 所有决策熵之和（可用于鼓励探索）
        """
        device = self.start_token.device
        h_t = torch.zeros(1, self.hidden_dim, device=device)
        c_t = torch.zeros(1, self.hidden_dim, device=device)
        x_t = self.start_token.unsqueeze(0)  # [1, embed_dim]

        # Step 1: 采样层数
        logits, h_t, c_t = self._step(x_t, h_t, c_t, self.num_layers_head)
        dist = Categorical(logits=logits)
        idx_num_layers = dist.sample()
        log_prob_sum = dist.log_prob(idx_num_layers)
        entropy_sum = dist.entropy()
        x_t = self.action_embed(idx_num_layers)

        # Step 2: 采样隐藏维度
        logits, h_t, c_t = self._step(x_t, h_t, c_t, self.hidden_size_head)
        dist = Categorical(logits=logits)
        idx_hidden_size = dist.sample()
        log_prob_sum = log_prob_sum + dist.log_prob(idx_hidden_size)
        entropy_sum = entropy_sum + dist.entropy()
        x_t = self.action_embed(idx_hidden_size)

        # Step 3: 采样激活函数
        logits, h_t, c_t = self._step(x_t, h_t, c_t, self.activation_head)
        dist = Categorical(logits=logits)
        idx_activation = dist.sample()
        log_prob_sum = log_prob_sum + dist.log_prob(idx_activation)
        entropy_sum = entropy_sum + dist.entropy()

        architecture = {
            "num_layers": self.search_space.num_layers_choices[idx_num_layers.item()],
            "hidden_size": self.search_space.hidden_size_choices[idx_hidden_size.item()],
            "activation": self.search_space.activation_choices[idx_activation.item()],
        }

        if not with_grad:
            # 推理时返回脱离计算图的值
            return architecture, log_prob_sum.detach(), entropy_sum.detach()

        return architecture, log_prob_sum, entropy_sum


class RLNAS:
    """
    使用 REINFORCE 训练 Controller 的 NAS 主流程。

    evaluator(architecture) -> reward(float)
    - 你可以在 evaluator 内部训练并验证子网络，然后返回验证准确率等指标作为 reward。
    """

    def __init__(
        self,
        controller: RNNController,
        evaluator: Callable[[Dict[str, object]], float],
        lr: float = 3e-4,
        entropy_coef: float = 1e-3,
        baseline_momentum: float = 0.9,
    ):
        self.controller = controller
        self.evaluator = evaluator
        self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.baseline_momentum = baseline_momentum
        self.baseline = None  # 指数滑动平均 baseline，用于降低方差

    def _update_baseline(self, reward: float) -> float:
        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward
        return self.baseline

    def search(self, episodes: int = 100) -> Tuple[Dict[str, object], float, List[Dict[str, object]]]:
        """
        执行 NAS 搜索。
        返回：
        - best_arch: 最优架构
        - best_reward: 最优奖励
        - history: 每轮日志（方便可视化和分析）
        """
        best_arch = None
        best_reward = float("-inf")
        history: List[Dict[str, object]] = []

        self.controller.train()

        for ep in range(1, episodes + 1):
            # 1) 从 Controller 采样一个架构
            arch, log_prob_sum, entropy_sum = self.controller.forward_and_sample(with_grad=True)

            # 2) 评估架构，得到 reward（标量）
            reward = float(self.evaluator(arch))

            # 3) 计算 advantage（reward - baseline）
            baseline = self._update_baseline(reward)
            advantage = reward - baseline

            # 4) REINFORCE 损失：-log_prob * advantage
            #    再加一个熵正则，鼓励探索
            policy_loss = -log_prob_sum * advantage
            entropy_bonus = self.entropy_coef * entropy_sum
            loss = policy_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if reward > best_reward:
                best_reward = reward
                best_arch = arch

            history.append(
                {
                    "episode": ep,
                    "architecture": arch,
                    "reward": reward,
                    "baseline": baseline,
                    "advantage": advantage,
                    "loss": float(loss.item()),
                }
            )

            print(
                f"[Episode {ep:03d}] reward={reward:.4f}, baseline={baseline:.4f}, "
                f"adv={advantage:.4f}, arch={arch}"
            )

        return best_arch, best_reward, history


# --------------------------- 示例评估函数（可替换） ---------------------------
def toy_evaluator(arch: Dict[str, object]) -> float:
    """
    一个玩具 reward 函数，用于演示流程。
    实际使用时请替换为“训练子网络并在验证集评估”的逻辑。
    """
    # 假设偏好：2-3 层、hidden_size=128、激活为 relu
    reward = 0.0
    reward += 1.0 if arch["num_layers"] in (2, 3) else 0.3
    reward += 1.2 if arch["hidden_size"] == 128 else 0.5
    reward += 0.8 if arch["activation"] == "relu" else 0.4

    # 加少量噪声，模拟训练波动
    reward += torch.randn(1).item() * 0.05
    return reward


if __name__ == "__main__":
    # 固定随机种子，保证可复现
    torch.manual_seed(42)

    search_space = SearchSpace()
    controller = RNNController(search_space)
    nas = RLNAS(controller, evaluator=toy_evaluator, lr=1e-3, entropy_coef=1e-3)

    best_arch, best_reward, _ = nas.search(episodes=50)
    print("\nBest Architecture:", best_arch)
    print("Best Reward:", round(best_reward, 4))

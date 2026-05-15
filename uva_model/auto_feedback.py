"""
C 方案：无标注自动反馈推断 via 行为信号。

三层信号：
1. 接纳信号（acceptance）：用户下一句提及模型的焦点词/联想 → +reward
2. 回避信号（avoidance）：用户回复很短 & 上轮是探针 → -reward  
3. 持续信号（continuation）：对话持续超过阈值轮数 → +reward

每层信号都返回可选的 float reward（若无信号则 None）；
最终通过加权平均综合多个信号。

所有 reward 强度都较弱（±0.3 以下），防止噪声累积；
依赖 EMA decay=0.98 长期纠正。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoFeedbackConfig:
    """自动反馈参数配置。"""

    enabled: bool = True
    #: 用户提及焦点词时的正反馈强度
    acceptance_reward: float = 0.5
    #: 检测回避时的负反馈强度
    avoidance_reward: float = -0.2
    #: 持续对话的最小轮数阈值
    continuation_min_turns: int = 3
    #: 持续对话的基础正反馈
    continuation_reward_base: float = 0.3
    #: 判定"敷衍回复"的最大字数
    evasion_max_length: int = 8
    #: 多个信号时，是否取平均（True）还是取最大（False）
    combine_mode: str = "mean"  # "mean" or "max"


def infer_acceptance_feedback(
    prev_focus_token: str,
    prev_association_pick: str,
    curr_user_text: str,
    reward_strength: float = 0.5,
) -> Optional[float]:
    """
    接纳信号：用户下一句提及了焦点词或联想词。
    
    Args:
        prev_focus_token: 上一轮的 conflict_focus_token
        prev_association_pick: 上一轮的 association_pick  
        curr_user_text: 本轮用户输入
        reward_strength: 正反馈强度
    
    Returns:
        若检测到接纳 → reward_strength，否则 None
    """
    if not curr_user_text.strip():
        return None
    
    # 检查焦点词
    focus = (prev_focus_token or "").strip()
    if focus and focus in curr_user_text:
        return float(reward_strength)
    
    # 检查联想词
    assoc = (prev_association_pick or "").strip()
    if assoc and assoc in curr_user_text:
        return float(reward_strength)
    
    return None


def infer_avoidance_feedback(
    prev_branch: str,
    prev_focus_token: str,
    curr_user_text: str,
    evasion_max_length: int = 8,
    reward_strength: float = -0.2,
) -> Optional[float]:
    """
    回避信号：上轮是探针，本轮回复很短且不提焦点词。
    
    Args:
        prev_branch: 上一轮输出分支
        prev_focus_token: 上一轮的 conflict_focus_token
        curr_user_text: 本轮用户输入
        evasion_max_length: 超过此长度就不算敷衍
        reward_strength: 负反馈强度
    
    Returns:
        若检测到回避 → reward_strength，否则 None
    """
    if not curr_user_text.strip():
        return None
    
    # 条件1：上轮是探针类
    is_probe = prev_branch in ("conflict_probe", "associative_probe")
    if not is_probe:
        return None
    
    # 条件2：本轮回复很短（敷衍）
    is_evasive = len(curr_user_text.strip()) <= evasion_max_length
    if not is_evasive:
        return None
    
    # 条件3：本轮没有提及上轮的焦点
    focus = (prev_focus_token or "").strip()
    has_focus = focus and focus in curr_user_text
    if has_focus:
        return None  # 用户虽然短回，但接住了焦点 → 不是回避
    
    return float(reward_strength)


def infer_continuation_feedback(
    consecutive_turns: int,
    min_turns: int = 3,
    base_reward: float = 0.3,
) -> Optional[float]:
    """
    持续信号：对话持续超过阈值 → 隐式认可。
    
    Args:
        consecutive_turns: 连续对话轮数
        min_turns: 触发阈值
        base_reward: 基础正反馈
    
    Returns:
        若连续轮数 >= min_turns → 衰减的正反馈，否则 None
    """
    if consecutive_turns < min_turns:
        return None
    
    # 随着轮数增加，反馈逐渐减弱（避免无限正反馈）
    # 例如：3轮 +0.1，4轮 +0.15，...，最多 +0.3
    excess = consecutive_turns - min_turns
    reward = base_reward * min(1.0, excess / 5.0)  # 5 轮后饱和
    return max(0.0, float(reward))


def combine_rewards(
    rewards: list[float],
    mode: str = "mean",
) -> Optional[float]:
    """
    综合多个反馈信号。
    
    Args:
        rewards: 若干反馈值列表
        mode: "mean" 取平均，"max" 取最强信号
    
    Returns:
        综合后的反馈值，或 None 若 rewards 为空
    """
    if not rewards:
        return None
    
    if mode == "mean":
        return sum(rewards) / len(rewards)
    elif mode == "max":
        # 找绝对值最大的（保留正负）
        return max(rewards, key=abs)
    else:
        return sum(rewards) / len(rewards)


def infer_auto_feedback(
    prev_branch: str,
    prev_focus_token: str,
    prev_association_pick: str,
    curr_user_text: str,
    consecutive_turns: int,
    config: AutoFeedbackConfig,
) -> Optional[float]:
    """
    综合三层行为信号，推断单个反馈值。
    
    Args:
        prev_branch: 上一轮输出分支
        prev_focus_token: 上一轮焦点词
        prev_association_pick: 上一轮联想词
        curr_user_text: 本轮用户输入
        consecutive_turns: 当前连续对话轮数
        config: 自动反馈配置
    
    Returns:
        推断出的反馈值（-1 到 1 之间），或 None 若无有效信号
    """
    if not config.enabled:
        return None
    
    rewards: list[float] = []
    
    # 层1：接纳信号
    acc = infer_acceptance_feedback(
        prev_focus_token,
        prev_association_pick,
        curr_user_text,
        reward_strength=config.acceptance_reward,
    )
    if acc is not None:
        rewards.append(acc)
    
    # 层2：回避信号
    avoid = infer_avoidance_feedback(
        prev_branch,
        prev_focus_token,
        curr_user_text,
        evasion_max_length=config.evasion_max_length,
        reward_strength=config.avoidance_reward,
    )
    if avoid is not None:
        rewards.append(avoid)
    
    # 层3：持续信号（此时 prev_branch 已经老了，实际上应该用 last_turn.output_branch）
    # 注：持续信号在 agent.turn() 内部独立处理，这里不重复
    
    return combine_rewards(rewards, mode=config.combine_mode)


__all__ = [
    "AutoFeedbackConfig",
    "infer_acceptance_feedback",
    "infer_avoidance_feedback",
    "infer_continuation_feedback",
    "combine_rewards",
    "infer_auto_feedback",
]

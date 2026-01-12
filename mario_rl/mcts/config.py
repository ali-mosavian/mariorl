"""
Configuration for MCTS exploration.
"""

from typing import Literal
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """
    Configuration for MCTS exploration.

    Attributes:
        num_simulations: Number of MCTS simulations per exploration call.
            More simulations = better action estimates but slower.
        max_rollout_depth: Maximum depth of rollout from leaf nodes.
            Deeper rollouts give better value estimates but are slower.
        exploration_constant: UCB exploration constant (c in UCB formula).
            Higher = more exploration, lower = more exploitation.
        discount: Discount factor for future rewards in rollouts.
        rollout_policy: How to select actions during rollouts.
            - "random": Pure random (fast but uninformed)
            - "policy": Use network policy (slower but better)
            - "mixed": Mix of both (balanced)
        policy_mix_ratio: When rollout_policy="mixed", fraction of policy vs random.
            0.7 means 70% policy actions, 30% random.
        value_source: How to estimate leaf node values.
            - "rollout": Full rollout to terminal (slow but accurate)
            - "network": Use value network directly (fast)
            - "mixed": Bootstrap rollout with network value
        min_visits_for_expansion: Minimum visits before expanding a node.
            Helps focus search on promising branches.
        use_prior: Whether to use policy network as prior for UCB.
            True enables PUCT (Predictor + UCB) like AlphaZero.
        prior_weight: Weight of prior in PUCT formula.
            Only used when use_prior=True.
        sequence_length: Number of actions to return in best_sequence.
            1 = return single best action (default behavior)
            >1 = return best action sequence from rollouts (like old MCTS)
    """

    num_simulations: int = 50
    max_rollout_depth: int = 20
    exploration_constant: float = 1.41
    discount: float = 0.99

    # Rollout policy
    rollout_policy: Literal["random", "policy", "mixed"] = "mixed"
    policy_mix_ratio: float = 0.7

    # Value estimation
    value_source: Literal["rollout", "network", "mixed"] = "mixed"

    # Advanced options
    min_visits_for_expansion: int = 0
    use_prior: bool = False
    prior_weight: float = 1.0

    # Action sequence settings (like old MCTS)
    # When > 1, MCTS returns the best action sequence found during rollouts
    # and the caller should execute the entire sequence before calling MCTS again
    sequence_length: int = 1  # 1 = single action (default), >1 = return sequence

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.max_rollout_depth < 1:
            raise ValueError("max_rollout_depth must be >= 1")
        if not 0 <= self.policy_mix_ratio <= 1:
            raise ValueError("policy_mix_ratio must be in [0, 1]")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")

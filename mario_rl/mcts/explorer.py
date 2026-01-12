"""
MCTS Explorer for training data collection.

Collects ALL transitions from ALL branches (good and bad paths)
for complete value landscape learning.
"""

from __future__ import annotations

import hashlib
from typing import Any
from typing import Callable
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

import numpy as np

from mario_rl.mcts.node import MCTSNode
from mario_rl.core.types import Transition
from mario_rl.mcts.config import MCTSConfig
from mario_rl.mcts.protocols import ValueAdapter
from mario_rl.mcts.protocols import PolicyAdapter
from mario_rl.mcts.protocols import WorldModelAdapter


def _hash_obs(obs: np.ndarray) -> str:
    """Hash observation for state deduplication."""
    return hashlib.md5(obs.tobytes()).hexdigest()


@dataclass
class ExplorationResult:
    """
    Result from MCTS exploration.

    Attributes:
        transitions: All transitions collected from all branches
        best_action: Best action from root based on visit counts
        root: Root node of the search tree (for analysis)
        stats: Exploration statistics
    """

    transitions: list[Transition]
    best_action: int
    root: MCTSNode
    stats: dict[str, Any]


@dataclass
class MCTSExplorer:
    """
    MCTS explorer that collects ALL transitions for training.

    Works with any RL algorithm through protocol-based adapters.
    Collects both successful and failed trajectories to learn
    complete value landscape.

    Two modes:
    1. Real MCTS: Uses environment save/restore (for training)
    2. Imagined MCTS: Uses world model (for inference without emulator)

    Attributes:
        config: MCTS configuration
        policy: Policy adapter for action selection
        value_fn: Value adapter for leaf evaluation
        num_actions: Number of available actions
        world_model: Optional world model for imagined rollouts
    """

    config: MCTSConfig
    policy: PolicyAdapter
    value_fn: ValueAdapter
    num_actions: int
    world_model: Optional[WorldModelAdapter] = None

    # State deduplication (like original MCTS)
    # Prevents expanding to already-visited states (avoids cycles)
    _visited_states: set[str] = field(init=False, default_factory=set)

    # Statistics (mutable)
    _total_simulations: int = field(init=False, default=0)
    _total_transitions: int = field(init=False, default=0)
    _total_explorations: int = field(init=False, default=0)

    def explore(
        self,
        env: Any,  # gym.Env with unwrapped.dump_state/load_state
        root_obs: np.ndarray,
        get_obs_fn: Optional[Callable[[Any], np.ndarray]] = None,
    ) -> ExplorationResult:
        """
        Run MCTS from current state, collect ALL transitions.

        This is the main entry point for training data collection.
        Uses real environment with save/restore.

        Args:
            env: Environment with save/restore capability
                Must have env.unwrapped.dump_state() and load_state()
            root_obs: Current observation (C, H, W)
            get_obs_fn: Optional function to get observation from env
                If None, uses env.unwrapped.screen

        Returns:
            ExplorationResult containing all transitions and best action
        """
        all_transitions: list[Transition] = []

        # Save root state
        root_snapshot = env.unwrapped.dump_state()

        # Create root node and mark as visited
        root = MCTSNode(
            state_snapshot=root_snapshot,
            obs=root_obs.copy(),
        )
        self._visited_states.add(_hash_obs(root_obs))

        # Set prior probabilities if using PUCT
        if self.config.use_prior:
            self.policy.get_action_probs(root_obs)
            root.prior = 1.0  # Root has uniform prior

        for _ in range(self.config.num_simulations):
            # Restore to root state
            env.unwrapped.load_state(root_snapshot)

            # Selection: traverse tree to leaf
            node, path_transitions = self._select(root, env, get_obs_fn)
            all_transitions.extend(path_transitions)

            if not node.terminal:
                # Expansion: add new child
                child, expand_transition = self._expand(node, env, get_obs_fn)
                if expand_transition:
                    all_transitions.append(expand_transition)

                if child and not child.terminal:
                    # Rollout: simulate to terminal or max depth
                    rollout_value, rollout_transitions = self._rollout(child, env, get_obs_fn)
                    all_transitions.extend(rollout_transitions)

                    # Backpropagation
                    self._backpropagate(child, rollout_value)
                elif child:
                    # Terminal node - get terminal value and backprop
                    terminal_value = self._get_terminal_value(child)
                    self._backpropagate(child, terminal_value)

            self._total_simulations += 1

        # Restore environment to original state
        env.unwrapped.load_state(root_snapshot)

        # Get best action from visit counts
        best_action = self._get_best_action(root)

        self._total_transitions += len(all_transitions)
        self._total_explorations += 1

        return ExplorationResult(
            transitions=all_transitions,
            best_action=best_action,
            root=root,
            stats={
                "simulations": self.config.num_simulations,
                "transitions_collected": len(all_transitions),
                "tree_depth": self._get_tree_depth(root),
                "tree_size": self._count_nodes(root),
            },
        )

    def explore_imagined(
        self,
        obs: np.ndarray,
    ) -> int:
        """
        Run imagined MCTS using world model (no environment needed).

        This is for inference when we don't have access to the emulator.
        Uses learned world model to predict transitions.

        Args:
            obs: Current observation

        Returns:
            Best action to take
        """
        if self.world_model is None:
            raise ValueError("World model required for imagined MCTS")

        # Encode observation to latent space
        latent = self.world_model.encode(obs)

        # Create root node (no snapshot needed for imagined MCTS)
        root = MCTSNode(
            state_snapshot=np.array([]),  # Not used
            obs=obs.copy(),
        )
        root_latent = latent

        for _ in range(self.config.num_simulations):
            # Selection in latent space
            node = root
            current_latent = root_latent.clone()

            while node.children and not node.terminal:
                node = node.best_child(
                    self.config.exploration_constant,
                    use_puct=self.config.use_prior,
                    prior_weight=self.config.prior_weight,
                )
                # Imagine step to update latent
                assert node.action is not None, "Child node must have an action"
                current_latent, _, done = self.world_model.imagine_step(current_latent, node.action)
                if done:
                    node.terminal = True
                    break

            if not node.terminal:
                # Expansion
                untried = node.get_untried_actions(self.num_actions)
                if untried:
                    action = self._select_expansion_action(node, untried)
                    next_latent, reward, done = self.world_model.imagine_step(current_latent, action)

                    # Decode for observation (optional, for node storage)
                    try:
                        next_obs = self.world_model.decode(next_latent)
                    except NotImplementedError:
                        next_obs = obs  # Fallback

                    child = MCTSNode(
                        state_snapshot=np.array([]),
                        obs=next_obs,
                        parent=node,
                        action=action,
                        terminal=done,
                    )
                    node.children.append(child)

                    # Get value estimate
                    value = self.value_fn.get_value(next_obs)
                    self._backpropagate(child, value)

        return self._get_best_action(root)

    def _select(
        self,
        node: MCTSNode,
        env: Any,
        get_obs_fn: Optional[Callable[[Any], np.ndarray]],
    ) -> tuple[MCTSNode, list[Transition]]:
        """
        Select leaf node using UCB/PUCT, collecting transitions.

        Traverses tree from root to a leaf node that needs expansion,
        replaying actions in the environment along the way.
        """
        transitions: list[Transition] = []

        while node.children and not node.terminal:
            if node.visits < self.config.min_visits_for_expansion:
                break

            # Select best child
            best_child = node.best_child(
                self.config.exploration_constant,
                use_puct=self.config.use_prior,
                prior_weight=self.config.prior_weight,
            )

            # Replay action in environment
            assert best_child.action is not None, "Child node must have an action"
            env.unwrapped.load_state(node.state_snapshot)
            next_obs, reward, done, truncated, info = env.step(best_child.action)

            # Get proper observation if needed
            if get_obs_fn:
                next_obs = get_obs_fn(env)

            transitions.append(
                Transition(
                    state=node.obs,
                    action=best_child.action,
                    reward=float(reward),
                    next_state=next_obs,
                    done=done or truncated,
                )
            )

            node = best_child
            if done or truncated:
                node.terminal = True
                break

        return node, transitions

    def _expand(
        self,
        node: MCTSNode,
        env: Any,
        get_obs_fn: Optional[Callable[[Any], np.ndarray]],
    ) -> tuple[Optional[MCTSNode], Optional[Transition]]:
        """
        Expand node by adding one new child for an untried action.

        Uses state deduplication to avoid expanding to already-visited states
        (like original MCTS), but still collects the transition for training.
        """
        if node.terminal:
            return None, None

        untried = node.get_untried_actions(self.num_actions)
        if not untried:
            return None, None

        # Select action for expansion
        action = self._select_expansion_action(node, untried)

        # Take action in environment
        env.unwrapped.load_state(node.state_snapshot)
        next_obs, reward, done, truncated, info = env.step(action)

        # Get proper observation
        if get_obs_fn:
            next_obs = get_obs_fn(env)

        # Always create the transition (for training data)
        transition = Transition(
            state=node.obs,
            action=action,
            reward=float(reward),
            next_state=next_obs,
            done=done or truncated,
        )

        # Check if state already visited (deduplication like original MCTS)
        obs_for_hash = next_obs.copy() if isinstance(next_obs, np.ndarray) else next_obs
        state_hash = _hash_obs(obs_for_hash)
        if state_hash in self._visited_states:
            # State already visited - return transition but no new node
            # This avoids cycles in the tree while still collecting training data
            return None, transition

        # Mark state as visited
        self._visited_states.add(state_hash)

        # Create child node
        child = MCTSNode(
            state_snapshot=env.unwrapped.dump_state(),
            obs=obs_for_hash,
            parent=node,
            action=action,
            terminal=done or truncated,
        )

        # Set prior if using PUCT
        if self.config.use_prior:
            probs = self.policy.get_action_probs(node.obs)
            child.prior = probs[action]

        node.children.append(child)

        return child, transition

    def _select_expansion_action(
        self,
        node: MCTSNode,
        untried: list[int],
    ) -> int:
        """Select which untried action to expand."""
        if self.config.rollout_policy == "random":
            return int(np.random.choice(untried))

        if self.config.rollout_policy in ("policy", "mixed"):
            # Use policy to guide expansion
            probs = self.policy.get_action_probs(node.obs)
            # Mask tried actions
            mask = np.zeros_like(probs)
            mask[untried] = probs[untried]
            if mask.sum() > 0:
                mask = mask / mask.sum()
                return int(np.random.choice(len(mask), p=mask))

        return int(np.random.choice(untried))

    def _rollout(
        self,
        node: MCTSNode,
        env: Any,
        get_obs_fn: Optional[Callable[[Any], np.ndarray]],
    ) -> tuple[float, list[Transition]]:
        """
        Rollout from node to estimate value and collect transitions.

        Uses policy/random mix for action selection based on config.
        """
        transitions: list[Transition] = []
        total_reward = 0.0
        discount = 1.0

        current_obs = node.obs
        env.unwrapped.load_state(node.state_snapshot)

        done = False
        truncated = False

        for _ in range(self.config.max_rollout_depth):
            # Select action
            action = self._select_rollout_action(current_obs)

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Get proper observation
            if get_obs_fn:
                next_obs = get_obs_fn(env)

            transitions.append(
                Transition(
                    state=current_obs,
                    action=action,
                    reward=float(reward),
                    next_state=next_obs,
                    done=done or truncated,
                )
            )

            total_reward += discount * float(reward)
            discount *= self.config.discount

            if done or truncated:
                break

            current_obs = next_obs

        # Bootstrap with network value for non-terminal states
        if not (done or truncated):
            if self.config.value_source in ("network", "mixed"):
                bootstrap_value = self.value_fn.get_value(current_obs)
                total_reward += discount * bootstrap_value

        return total_reward, transitions

    def _select_rollout_action(self, obs: np.ndarray) -> int:
        """Select action during rollout based on config."""
        if self.config.rollout_policy == "random":
            return int(np.random.randint(self.num_actions))

        if self.config.rollout_policy == "policy":
            return self.policy.get_action(obs)

        # Mixed policy
        if np.random.random() < self.config.policy_mix_ratio:
            return self.policy.get_action(obs)
        return int(np.random.randint(self.num_actions))

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate value through tree to root.

        Note: We do NOT discount during backprop (same as standard MCTS).
        The value already includes discounting from the rollout phase.
        All nodes in the path get equal credit for the outcome.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _get_terminal_value(self, node: MCTSNode) -> float:
        """Get value for terminal node."""
        # For terminal states, value comes from the final reward
        # The node's observation represents the terminal state
        # Could use network value or just return 0
        return 0.0

    def _get_best_action(self, root: MCTSNode) -> int:
        """Get best action from root based on visit counts."""
        if not root.children:
            # No children - use policy
            return self.policy.get_action(root.obs)

        # Return action of most visited child
        best_child = root.most_visited_child()
        assert best_child.action is not None, "Best child must have an action"
        return best_child.action

    def _get_tree_depth(self, node: MCTSNode) -> int:
        """Get maximum depth of tree."""
        if not node.children:
            return 0
        return 1 + max(self._get_tree_depth(c) for c in node.children)

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def get_stats(self) -> dict[str, Any]:
        """Get exploration statistics."""
        return {
            "total_simulations": self._total_simulations,
            "total_transitions": self._total_transitions,
            "total_explorations": self._total_explorations,
            "avg_transitions_per_explore": (self._total_transitions / max(1, self._total_explorations)),
        }

    def reset_stats(self) -> None:
        """Reset exploration statistics."""
        self._total_simulations = 0
        self._total_transitions = 0
        self._total_explorations = 0

    def clear_visited_states(self) -> None:
        """Clear visited states cache (for starting fresh exploration)."""
        self._visited_states.clear()

    def reset(self) -> None:
        """Full reset: clear stats and visited states."""
        self.reset_stats()
        self.clear_visited_states()

import math
import random
import hashlib
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple
from typing import Optional

import numpy as np
import gymnasium as gym


@dataclass
class MCTSNode:
    state_dump: np.ndarray
    state_hash: str
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    visits: int = 0
    reward: float = 0.0
    children: List['MCTSNode'] = field(default_factory=list)
    terminal: bool = False

    def is_fully_expanded(self, action_space_size: int) -> bool:
        '''
        Check if all possible actions have been expanded for this node.
        '''
        return len(self.children) == action_space_size

    def best_child(self, exploration_param: float = 1.41) -> 'MCTSNode':
        '''
        Select the best child node using the UCT formula.
        '''
        return max(
            self.children,
            key=lambda child: uct_value(
                total_visits=self.visits,
                node_wins=child.reward,
                node_visits=child.visits,
                exploration_param=exploration_param,
            ),
        )

    def find_child_with_action(self, action: int) -> Optional['MCTSNode']:
        '''
        Find child node that corresponds to a specific action.
        '''
        for child in self.children:
            if child.action == action:
                print(f'Found child with action {action}')
                return child
        return None


def hash_state(state: np.ndarray) -> str:
    '''
    Generates a hash for the given np.ndarray state.
    '''
    return hashlib.md5(state.tobytes()).hexdigest()


def uct_value(
    total_visits: int,
    node_wins: float,
    node_visits: int,
    exploration_param: float = 1.41
) -> float:
    '''
    Calculates the UCT value for a node.
    '''
    if node_visits == 0:
        return float('inf')
    return (node_wins / node_visits) + exploration_param * math.sqrt(
        math.log(total_visits) / node_visits
    )


class MCTS:
    def __init__(self) -> None:
        self.visited_states: set[str] = set()
        self.root: Optional[MCTSNode] = None

    def is_state_visited(self, image: np.ndarray) -> bool:
        '''
        Checks if a state has been visited by comparing its image hash.
        '''
        return hash_state(image) in self.visited_states

    def select(
        self,
        node: MCTSNode,
        env: gym.Env,
        action_space_size: int
    ) -> MCTSNode:
        '''
        Traverse the tree using UCT until a node is found for expansion.
        During traversal, keep environment state in sync with selected child nodes.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()

        while node.children and node.is_fully_expanded(action_space_size) and not node.terminal:
            node = node.best_child()
            env.unwrapped.load_state(node.state_dump)
            if node.action is not None:
                _, _, terminated, truncated, _ = env.step(node.action)
                if terminated or truncated:
                    node.terminal = True
                    break

        env.unwrapped.load_state(snapshot)
        return node

    def expand(
        self,
        node: MCTSNode,
        env: gym.Env,
        action_space_size: int
    ) -> Optional[MCTSNode]:
        '''
        Expand a node by creating a new child for an untried action,
        skipping already-visited states.
        '''
        if node.terminal:
            return None

        untried_actions: List[int] = [
            a for a in range(action_space_size)
            if all(child.action != a for child in node.children)
        ]
        if not untried_actions:
            return None

        action: int = random.choice(untried_actions)
        snapshot: np.ndarray = env.unwrapped.dump_state()
        state: np.ndarray
        reward: float
        terminated: bool
        truncated: bool
        state, reward, terminated, truncated, _ = env.step(action)

        child_terminal: bool = False
        if terminated or truncated:
            child_terminal = True

        state_hash: str = hash_state(state)
        if state_hash in self.visited_states:
            env.unwrapped.load_state(snapshot)
            return None

        self.visited_states.add(state_hash)

        child_node: MCTSNode = MCTSNode(
            state_dump=env.unwrapped.dump_state(),
            state_hash=state_hash,
            parent=node,
            action=action,
            terminal=child_terminal
        )
        node.children.append(child_node)
        env.unwrapped.load_state(snapshot)

        return child_node

    def simulate(
        self,
        env: gym.Env,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform a simulation (rollout) to estimate the reward and find good action sequences.
        Uses a random rollout policy by default.
        '''
        actions: List[int] = []
        total_reward: float = 0.0
        snapshot: np.ndarray = env.unwrapped.dump_state()

        for _ in range(max_trajectory_length):
            action: int = int(env.action_space.sample())
            _, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            total_reward += reward
            if terminated or truncated:
                #print(f'Terminal state, reward: {reward}')
                break

        env.unwrapped.load_state(snapshot)
        return total_reward, actions
    
    def simulate2(
        self,
        env: gym.Env,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform a simulation (rollout) to estimate the reward and find good action sequences,
        delegating to a recursive helper function for clarity.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()

        total_reward, actions = self._simulate_recursive(
            env=env,
            step=0,
            max_steps=max_trajectory_length,
            snapshot=snapshot,
            actions=[],
            total_reward=0.0,
            best_partial_reward=0.0,
            best_actions=[]
        )

        env.unwrapped.load_state(snapshot)
        return total_reward, actions

    def _simulate_recursive(
        self,
        env: gym.Env,
        step: int,
        max_steps: int,
        snapshot: np.ndarray,
        actions: List[int],
        total_reward: float,
        best_partial_reward: float,
        best_actions: List[int]
    ) -> Tuple[float, List[int]]:
        '''
        A recursive helper function for simulation.
        If the agent reaches a terminal state, it backtracks to the best partial path discovered so far.
        '''
        if step >= max_steps:
            return total_reward, actions

        action: int = int(env.action_space.sample())
        _, reward, terminated, truncated, _ = env.step(action)

        updated_actions: List[int] = [*actions, action]
        updated_reward: float = total_reward + reward

        updated_best_partial_reward: float = best_partial_reward
        updated_best_actions: List[int] = [*best_actions]

        if updated_reward > best_partial_reward:            
            updated_best_partial_reward = updated_reward
            updated_best_actions = updated_actions.copy()
            if not (terminated or truncated):
                snapshot = env.unwrapped.dump_state()

        if terminated or truncated:
            env.unwrapped.load_state(snapshot)
            return updated_best_partial_reward, updated_best_actions

        return self._simulate_recursive(
            env=env,
            step=step + 1,
            max_steps=max_steps,
            snapshot=snapshot,
            actions=updated_actions,
            total_reward=updated_reward,
            best_partial_reward=updated_best_partial_reward,
            best_actions=updated_best_actions
        )    

    def backpropagate(
        self,
        node: MCTSNode,
        reward: float
    ) -> None:
        '''
        Backpropagate the reward to all ancestors.
        '''
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def search(
        self,
        env: gym.Env,
        root: MCTSNode,
        num_simulations: int,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform MCTS to find the best action sequence.
        '''
        action_space_size: int = env.action_space.n

        best_sequence: List[int] = []
        best_reward: float = float('-inf')
        
        for _ in range(num_simulations):
            node: MCTSNode = root
            snapshot: np.ndarray = env.unwrapped.dump_state()

            for depth in range(max_trajectory_length):
                node = self.select(node, env, action_space_size)
                new_node: Optional[MCTSNode] = self.expand(node, env, action_space_size)
                if not new_node:
                    break

                rollout_reward, roll_actions = self.simulate(env, max_trajectory_length - depth)

                if rollout_reward > best_reward:
                    best_reward = rollout_reward
                    best_sequence = roll_actions

                self.backpropagate(new_node, rollout_reward)



            env.unwrapped.load_state(snapshot)

        #best_sequence: List[int] = []
        #current_node: MCTSNode = root
        #while current_node.children:
        #    current_node = max(current_node.children, key=lambda child: child.reward)
        #    if current_node.action is not None:
        #        best_sequence.append(current_node.action)

        return best_reward, best_sequence

    def update_root_after_action(self, action: int, env: gym.Env) -> None:
        '''
        Update the root node after an action is taken.
        '''
        if self.root is None:
            return

        # Find the child node corresponding to the action
        new_root: Optional[MCTSNode] = self.root.find_child_with_action(action)
        
        if new_root is not None:
            # Detach from parent and update as new root
            new_root.parent = None
            self.root = new_root
        else:
            # If we can't find the child (shouldn't happen in normal use), create new root
            state_hash: str = hash_state(env.unwrapped.screen)
            self.root = MCTSNode(
                state_dump=env.unwrapped.dump_state(),
                state_hash=state_hash
            )

    def do_rollout(
        self,
        env: gym.Env,
        num_simulations: int = 10,
        max_trajectory_length: int = 25
    ) -> Tuple[float, List[int]]:
        '''
        Perform a rollout using MCTS, reusing existing tree if available.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()
        
        # Create root node only if we don't have one
        if self.root is None:
            initial_state: np.ndarray = env.unwrapped.screen
            self.root = MCTSNode(
                state_dump=snapshot,
                state_hash=hash_state(initial_state)
            )

        best_reward, best_sequence = self.search(
            env=env,
            root=self.root,
            num_simulations=num_simulations,
            max_trajectory_length=max_trajectory_length
        )
        env.unwrapped.load_state(snapshot)

        return best_reward, best_sequence
    



# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .pyenv
#     language: python
#     name: python3
# ---

# %%
import time
from typing import Union
from typing import Literal
from typing import Callable
from typing import Optional
from typing import NamedTuple
from dataclasses import field
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gym_super_mario_bros.mcts import MCTS


# %%
@dataclass(init=False)
class LazyFrame:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """
    shape: tuple[int, ...]
    dtype: np.dtype
    _hash: int
    _frame: Union[np.ndarray, bytes]
    _decoder: Union[Callable[[bytes], np.ndarray], Callable[[np.ndarray], np.ndarray]]

    def __init__(self, frame: np.ndarray, compression: Union[None, Literal['fast', 'high']] = None):
        """Lazyframe for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """        
        from hashlib import md5
        self.shape = frame.shape
        self.dtype = frame.dtype        
        self._frame = frame
        self._hash = int.from_bytes(md5(frame.tobytes()).digest(), 'little')
        self._decoder = lambda x: x
        

        if compression is None:
            return
        
        try:
            import lz4.block as lz4
        except ImportError:
            raise DependencyNotInstalled(
                "lz4 is not installed, run `pip install lz4`"
            )
        
        match compression:
            case 'fast':
                mode = 'fast'
            case 'high':
                mode = 'high_compression'
            case _:
                raise ValueError(f"Invalid compression mode: {compression}")
        
        self._frame = lz4.compress(frame.tobytes(), mode=mode)
        self._decoder = lambda x: (
            np.frombuffer(
                lz4.decompress(x),                 
                dtype=frame.dtype
            )
            .reshape(frame.shape)
        )

    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"LazyFrame(shape={self.shape}, hash={self._hash:x}, dtype={self.dtype})"
        

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.

        Args:
            dtype: The dtype of the stacked frames

        Returns:
            The array of stacked frames with dtype
        """
        return self._decoder(self._frame)

    def __len__(self):
        """Returns the number of frame stacks.

        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other
    

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, render_frames: bool = False, playback_speed: float = 1.0):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.render_frames = render_frames
        self.next_render_time = None
        self.playback_speed = playback_speed

    def step(self, action):
        """Repeat action, and sum reward"""
        done = False
        total_reward = 0.0

        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r
            if done or truncated:
                break

            if self.render_frames:
                self.next_render_time = self.next_render_time or time.time()
                while time.time() < self.next_render_time:
                    time.sleep(1/self.metadata['render_fps']/2)
                self.next_render_time += 1/self.metadata['render_fps']/self.playback_speed
                self.env.render()

        return s_, total_reward, done, truncated, info


class Observation(NamedTuple):
    state: LazyFrame
    action: int    
    next_state: LazyFrame
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass(init=False)
class ObservationRecorder(gym.Wrapper):
    """Record the observations of the environment"""    
    action_space: gym.spaces.Discrete
    observations: list[Observation] = field(default_factory=list)
    record: bool = True
    

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observations = list()
        self.action_space = env.action_space


    def step(self, action: int):
        state = LazyFrame(self.env.unwrapped.screen.copy(), compression='high')        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_lf = LazyFrame(next_state.copy(), compression='high')
        
        
        if self.record:            
            self.observations.append(
                Observation(
                    state=state,
                    action=action, 
                    next_state=next_state_lf, 
                    reward=reward, 
                    terminated=terminated, 
                    truncated=truncated, 
                    info=info
                )
            )
            
        return next_state, reward, terminated, truncated, info


# %%
from mcts import MCTS
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

env = gym.make('SuperMarioBros-1-1-v0')

actions = SIMPLE_MOVEMENT
env = JoypadSpace(env, actions)


class Transition(NamedTuple):
    state: LazyFrame
    action: Optional[int] = None    
    reward: Optional[float] = None
    terminal: Optional[bool] = None


transitions = []

try:
    mcts = MCTS()
    env = SkipFrame(env, skip=8, render_frames=False, playback_speed=1)
    env = ObservationRecorder(env)

    done = True    
    actions = []

    last_state = None
    for _ in range(1000):
        if done:
            last_state, _ = env.reset()
            transitions.append(Transition(state=LazyFrame(last_state, compression='high')))

        #env.record = False

        if len(actions) == 0:
            reward, actions = mcts.do_rollout(env, num_simulations=5, max_trajectory_length=15)
            print(reward, actions)
            actions = actions

        if len(actions) == 0:
            actions = [env.action_space.sample()]

        #env.record = True
        env.render_frames = True
        env.next_render_time = None
        action, *actions = actions
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated   
        env.render_frames = False
        
        transitions.append(Transition(action=action, state=LazyFrame(state, compression='high'), reward=reward, terminal=done))
        last_state = state

except KeyboardInterrupt:
    pass

# %%
env.observations[0]

# %%
from PIL import Image
from collections import defaultdict

counter = defaultdict(int)

for t in env.observations:
    if t.state is not None:
        counter[hash(t.state)] += 1
    if t.next_state is not None:
        counter[hash(t.next_state)] += 1

sorted(counter.items(), key=lambda x: x[1], reverse=True)


# %%
t.state, t.next_state


# %%
env.reset();
for i in range(4):
    state, _, _, _, _ = env.step(env.action_space.sample())

snapshot = env.unwrapped.dump_state()
for i in range(4):
    _, _, _, _, _ = env.step(env.action_space.sample())

env.unwrapped.load_state(snapshot)
state2 = env.unwrapped.screen

for i in range(4):
    _, _, _, _, _ = env.step(env.action_space.sample())

env.unwrapped.load_state(snapshot)
state3 = env.unwrapped.screen




# %%
(hash(state.tobytes()) == hash(state2.tobytes())), (hash(state.tobytes()) == hash(state3.tobytes()))


# %%
@dataclass(frozen=True)
class Transition:
    action: int
    node: 'Node'
    reward: Optional[float] = None
    terminal: Optional[bool] = None

    def __hash__(self) -> int:
        return hash(hash(self.action) + hash(self.node)+hash(self.reward)+hash(self.terminal))

@dataclass
class Node:
    state: LazyFrame    
    parents: list['Node'] = field(default_factory=list)
    children: list[Transition] = field(default_factory=list)

    def __hash__(self) -> int:
        # Hash the actual numpy array contents
        return hash(self.state.__array__().tobytes())

    def __eq__(self, other: 'Node') -> bool:
        # Compare actual numpy arrays
        return np.array_equal(self.state.__array__(), other.state.__array__())

def build_trajectories(observations: list[Observation]) -> list[Node]:
    """Builds a trajectory tree from a list of observations.
    
    Args:
        observations: List of environment observations
        
    Returns:
        List of root nodes in the trajectory tree
    """
    seen_states: dict[int, Node] = {}
    root_nodes: list[Node] = []
    
    if not observations:
        return root_nodes
        
    # Initialize with first state
    root = Node(state=observations[0].state)
    root_nodes.append(root)
    seen_states[hash(root)] = root

    for observation in observations:
        state_hash = hash(observation.state.__array__().tobytes())
        next_state_hash = hash(observation.next_state.__array__().tobytes())
        
        # Get or create current state node
        if state_hash not in seen_states:
            state_node = Node(state=observation.state)
            root_nodes.append(state_node)
            seen_states[state_hash] = state_node
        else:
            state_node = seen_states[state_hash]

        # Get or create next state node
        if next_state_hash not in seen_states:
            next_state_node = Node(state=observation.next_state)
            seen_states[next_state_hash] = next_state_node
        else:
            next_state_node = seen_states[next_state_hash]

        # Connect nodes
        t = Transition(action=observation.action, node=next_state_node, reward=observation.reward, terminal=observation.terminated)
        if t not in state_node.children:
            state_node.children.append(t)
        if state_node not in next_state_node.parents:
            next_state_node.parents.append(state_node)
        
    return root_nodes


def print_trajectory(root: Node, max_depth: int = 10) -> None:
    """Prints the tree structure starting from the root node.
    
    Args:
        root: The root node to start printing from
        max_depth: Maximum depth to print to avoid infinite recursion
    """
    visited: set[Node] = set()
    
    def _print_node(node: Node, action: Optional[int] = None, depth: int = 0) -> None:
        if depth > max_depth or node in visited:
            return
        
        #visited.add(node)
        prefix = '  ' * depth
        node_id = f'node: {node.state._hash:x}'
        action_str = f' action: {action} -> {node_id}' if action is not None else node_id
        reward_str = f' (R: {node.reward:.2f})' if hasattr(node, 'reward') else ''
        term_str = ' [TERMINAL]' if getattr(node, 'terminated', False) else ''
        
        print(f'{prefix}{action_str}{reward_str}{term_str}')
        
        for transition in node.children:
            #print(transition)
            _print_node(transition.node, transition.action, depth + 1)

    _print_node(root)
    print(f'\nTotal unique nodes visited: {len(visited)}')



# %%

roots = build_trajectories(env.observations)
print_trajectory(roots[2])


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

a = env.observations[0].action
f1 = env.observations[0].state.__array__()
f2 = env.observations[1].next_state.__array__()

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.act(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.act(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.act(self.conv3(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        
        return x

class JEPAEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.cnn = CNNEncoder()
        self.mlp = nn.LazyLinear(embedding_dim)
        self.act = nn.GELU()

    def forward(self, emb: torch.Tensor, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.cnn(obs)
        x = x.flatten(1)
        x = torch.cat([x, emb, action], dim=1)
        x = self.act(self.mlp(x))
        return x

class IJEPAEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.cnn = CNNEncoder()
        self.mlp = nn.LazyLinear(embedding_dim)
        self.act = nn.GELU()

    def forward(self,obs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(obs)
        x = x.flatten(1)
        x = self.act(self.mlp(x))
        return x        


class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the flattened input back to spatial dimensions
        batch_size = x.shape[0]
        #x = x.view(batch_size, 128, 30, 32)  # Adjust dimensions based on your encoder's output
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        
        # Match original image format
        return x

class IJEPADecoder(nn.Module):
    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Linear(embedding_dim, 128*30*32)  
        self.cnn = CNNDecoder()
        self.act = nn.GELU()
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mlp(embedding))
        x = x.reshape(x.shape[0], 128, 30, 32)
        x = self.cnn(x)        
        return x

# Loss function for training
def jepa_loss(predicted_obs: torch.Tensor, target_obs: torch.Tensor) -> torch.Tensor:
    """Calculate RMSE loss between predicted and target observations."""
    return torch.sqrt(torch.mean((predicted_obs - target_obs) ** 2))




def ssim_loss(pred: torch.Tensor, 
              target: torch.Tensor,
              window_size: int = 11,
              sigma: float = 1.5) -> torch.Tensor:
    """Computes SSIM-based loss between images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        window_size: Size of the gaussian window
        sigma: Standard deviation of gaussian window
        
    Returns:
        1 - SSIM (as a loss value)
    """
    # Create gaussian window
    gaussian = torch.exp(
        torch.tensor(
            [-(x - window_size//2)**2 / float(2*sigma**2) 
             for x in range(window_size)]
        )
    )
    gaussian = gaussian / gaussian.sum()
    window = gaussian.unsqueeze(1) @ gaussian.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(pred.size(1), 1, window_size, window_size)
    window = window.to(pred.device)

    # Calculate SSIM
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2

    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return 1 - ssim.mean()


def psnr_loss(pred: torch.Tensor, 
              target: torch.Tensor, 
              max_val: Union[float, int] = 1.0,
              reduction: str = 'mean') -> torch.Tensor:
    """Computes PSNR (Peak Signal-to-Noise Ratio) between images.
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        max_val: Maximum value of the signal (1.0 if image is normalized, 255 if not)
        reduction: 'mean', 'sum', or 'none' to return per image PSNR
        
    Returns:
        PSNR loss value (lower is worse). For optimization, often used as -PSNR
    """
    # Ensure inputs are float tensors
    pred = pred.float()
    target = target.float()
    
    # Convert max_val to tensor and move to same device as input
    max_val = torch.tensor(max_val, device=pred.device, dtype=torch.float)
    
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    
    # Avoid log of zero
    eps = torch.finfo(torch.float32).eps
    mse = torch.clamp(mse, min=eps)
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
    
    # Apply reduction
    if reduction == 'mean':
        return psnr.mean()
    elif reduction == 'sum':
        return psnr.sum()
    else:  # 'none'
        return psnr

from typing import Union, Literal
import torch

def combined_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    max_val: float = 1.0,
    normalize_psnr: bool = True,
    ssim_window_size: int = 16,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> torch.Tensor:
    """Combines SSIM and PSNR losses with weighted importance.
    
    Args:
        pred: Predicted images (B,C,H,W) in range [0,1]
        target: Target images (B,C,H,W) in range [0,1]
        alpha: Weight for SSIM loss (1-alpha will be PSNR weight)
        max_val: Maximum value of input tensors
        normalize_psnr: Whether to normalize PSNR to similar scale as SSIM
        reduction: Reduction method for batch of images
        
    Returns:
        Combined loss value
    """
    # Calculate SSIM loss (ranges 0-1)
    ssim_loss_val = ssim_loss(pred, target)
    
    # Calculate PSNR loss
    psnr_val = psnr_loss(pred, target, max_val=max_val)
    
    if normalize_psnr:
        # Normalize PSNR to 0-1 range (assuming typical PSNR range of 0-50 dB)
        psnr_loss_val = 1.0 - (torch.clamp(psnr_val, 0, 50) / 50.0)
    else:
        # Use negative PSNR as loss
        psnr_loss_val = -psnr_val / 50.0  # Divide by 50 to balance magnitude
    
    # Combine losses
    combined_loss = alpha * ssim_loss_val + (1 - alpha) * psnr_loss_val
    
    # Apply reduction
    if reduction == 'mean':
        return combined_loss.mean()
    elif reduction == 'sum':
        return combined_loss.sum()
    return combined_loss

# Alternative version with dynamic weighting
def adaptive_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_start: float = 0.8,
    alpha_end: float = 0.2,
    current_epoch: int = 0,
    total_epochs: int = 100
) -> torch.Tensor:
    """Combines SSIM and PSNR losses with dynamic weighting over training.
    
    Args:
        pred: Predicted images
        target: Target images
        alpha_start: Initial SSIM weight
        alpha_end: Final SSIM weight
        current_epoch: Current training epoch
        total_epochs: Total training epochs
        
    Returns:
        Combined loss value
    """
    # Calculate dynamic alpha
    alpha = alpha_start + (alpha_end - alpha_start) * (current_epoch / total_epochs)
    
    return combined_quality_loss(pred, target, alpha=alpha)        

def psnr_loss_for_training(pred: torch.Tensor, 
                          target: torch.Tensor, 
                          max_val: Union[float, int] = 1.0) -> torch.Tensor:
    """Negative PSNR loss for training (since we want to maximize PSNR).
    
    Args:
        pred: Predicted images tensor of shape (B, C, H, W)
        target: Target images tensor of shape (B, C, H, W)
        max_val: Maximum value of the signal (1.0 if image is normalized, 255 if not)
        
    Returns:
        Negative PSNR value (for minimization during training)
    """
    return -psnr_loss(pred, target, max_val)    

class IJEPAModel(nn.Module):    
    def __init__(self, num_actions: int, embedding_dim: int = 1024):
        super().__init__()        
        self.encoder = IJEPAEncoder(embedding_dim=embedding_dim)
        self.decoder = IJEPADecoder(embedding_dim=embedding_dim)
        
    def loss(self, h: torch.Tensor, x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        """Calculate RMSE loss between predicted and target observations.
        
        Args:
            h: Tensor of shape (batch_size, embedding_dim)
            x: Predicted observations 
            xp: Target observations
            
        Returns:
            Loss including RMSE and correlation between samples
        """
        # Calculate RMSE loss
        #reconstruction_loss = psnr_loss_for_training(x, xp, max_val=1.0)
        reconstruction_loss = ssim_loss(x, xp, window_size=16, sigma=1.5)
        # Center the entire h matrix
        centered = h - h.mean(dim=0, keepdim=True)
        
        # Calculate correlation matrix between samples
        correlation = centered @ centered.T
        
        # The correlation matrix should be close to identity
        batch_size = h.size(0)
        identity = torch.eye(batch_size, device=h.device)
        
        corr_loss = torch.norm(correlation - identity)
        
        # Combine losses with a weighting factor
        lambda_corr = 0.1
        total_loss = reconstruction_loss #+ lambda_corr * corr_loss
        
        return total_loss


    def forward(self, x, return_loss: bool = False) -> torch.Tensor:        
        h = self.encoder(x)        
        xp = self.decoder(h)
        
        if return_loss:
            return xp, h, self.loss(h, x, xp)
        else:
            return xp, h





# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def available_device() -> torch.device:
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    return torch.device('cuda' if has_cuda else 'mps' if has_mps else 'cpu')

def train(x: torch.Tensor, model: nn.Module, num_epochs: int = 100, batch_size: int = 32) -> list[float]:
    """Training loop for the model.
    
    Args:
        x: Input tensor of shape [2886, 240, 256, 3]
        model: Neural network model
        num_epochs: Number of training epochs
    
    Returns:
        List of losses per epoch
    """
    
    # Create dataset and dataloader
    dataset = TensorDataset(torch.tensor(x/255.0, dtype=torch.float))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    losses: list[float] = []
    device = torch.device(available_device())
    model = model.to(device)
    
    #epoch_bar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', ncols=100)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        batch_bar = tqdm(dataloader, desc='Batch', unit='batch', leave=False, ncols=100)
        for batch in batch_bar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass            
            x = batch[0].to(device)
            xp, h, loss = model(x, return_loss=True)            
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1

            del x
            del xp
            del loss            
            batch_bar.set_postfix(loss=batch_loss)

        
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')
        # Print progress
        
    
    return losses

# Example usage:

states = dict()
for s in [*[o.state for o in env.observations], *[o.next_state for o in env.observations]]:
    states[hash(s)] = s

x = np.array([np.array(s) for s in states.values()])
x = x.transpose(0, 3, 1, 2)
print(x.shape)

model = IJEPAModel(num_actions=env.action_space.n, embedding_dim=1024)
losses = train(x, model)




# %%
x = np.array([np.array(s) for s in states.values()])
x = x.transpose(0, 3, 1, 2)

x1 = x[:32]
with torch.no_grad():
    x2, h = model(torch.tensor(x1/255.0, dtype=torch.float, device=available_device()))
    x2 = (x2*255.0).cpu().numpy()
    h = h.cpu().numpy()

x1 = x1.transpose(0, 2, 3, 1)
x2 = x2.transpose(0, 2, 3, 1)

x1.shape
x2.shape


# %%
x

# %%
h.sum(axis=1).shape


# %%
from PIL import Image
from IPython.display import display

display(Image.fromarray(np.uint8(x1[23])))
display(Image.fromarray(np.uint8(x2[23])))


# %%
states = dict()
for s in [*[o.state for o in env.observations], *[o.next_state for o in env.observations]]:
    states[hash(s)] = s

len(states)

# %%

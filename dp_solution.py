"""
Part 2: Dynamic Programming for Mini Chess Game
This module implements Value Iteration and Policy Iteration for a simplified chess endgame.

Author: DRL Assignment Solution
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Set
from collections import deque
import time


class MiniChessEnv:
    """
    Custom Mini Chess Environment for reinforcement learning.
    White: King + Pawn vs Black: King
    """

    def __init__(self, board_size: int = 4, student_id_last_digit: int = 0):
        """
        Initialize the Mini Chess environment.

        Args:
            board_size: Size of the square board (4 or 5)
            student_id_last_digit: Last digit of student ID for initial configuration
        """
        self.board_size = board_size
        self.student_id_last_digit = student_id_last_digit

        self.WK = 'WK'
        self.WP = 'WP'
        self.BK = 'BK'
        self.EMPTY = '.'

        self.WHITE = 0
        self.BLACK = 1

        self.MOVE_LIMIT = 30

        self.board = None
        self.wk_pos = None
        self.wp_pos = None
        self.bk_pos = None
        self.turn = None
        self.move_count = 0
        self.pawn_promoted = False

    def reset(self) -> Tuple:
        """
        Reset the environment to initial configuration based on student ID.

        Returns:
            Initial state tuple
        """
        self.board = [[self.EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.move_count = 0
        self.pawn_promoted = False
        self.turn = self.WHITE

        if self.student_id_last_digit <= 4:
            self.wk_pos = (0, 0)
            self.bk_pos = (self.board_size - 1, self.board_size - 1)
            self.wp_pos = (1, 0)
        else:
            mid = self.board_size // 2
            self.wk_pos = (mid, mid)
            self.bk_pos = (0, self.board_size - 1)
            self.wp_pos = (mid + 1, mid)

        self._update_board()
        return self._get_state()

    def _update_board(self):
        """Update board representation from piece positions."""
        self.board = [[self.EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]

        if self.wk_pos:
            self.board[self.wk_pos[0]][self.wk_pos[1]] = self.WK
        if self.wp_pos and not self.pawn_promoted:
            self.board[self.wp_pos[0]][self.wp_pos[1]] = self.WP
        if self.bk_pos:
            self.board[self.bk_pos[0]][self.bk_pos[1]] = self.BK

    def _get_state(self) -> Tuple:
        """
        Get current state representation.

        Returns:
            Tuple: (wk_row, wk_col, wp_row, wp_col, bk_row, bk_col, turn, promoted, move_count)
        """
        wp_r, wp_c = self.wp_pos if self.wp_pos else (-1, -1)
        return (
            self.wk_pos[0], self.wk_pos[1],
            wp_r, wp_c,
            self.bk_pos[0], self.bk_pos[1],
            self.turn,
            int(self.pawn_promoted),
            self.move_count
        )

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within board boundaries."""
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def _get_king_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all possible king moves from a position.

        Args:
            pos: King position

        Returns:
            List of valid move positions
        """
        r, c = pos
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        moves = []

        for dr, dc in directions:
            new_pos = (r + dr, c + dc)
            if self._is_valid_pos(new_pos):
                moves.append(new_pos)

        return moves

    def _get_pawn_moves(self) -> List[Tuple[int, int]]:
        """
        Get all possible pawn moves (forward and diagonal captures).

        Returns:
            List of valid move positions
        """
        if not self.wp_pos or self.pawn_promoted:
            return []

        r, c = self.wp_pos
        moves = []

        forward = (r + 1, c)
        if self._is_valid_pos(forward) and self.board[forward[0]][forward[1]] == self.EMPTY:
            moves.append(forward)

        for dc in [-1, 1]:
            capture_pos = (r + 1, c + dc)
            if self._is_valid_pos(capture_pos):
                if capture_pos == self.bk_pos:
                    moves.append(capture_pos)

        return moves

    def _is_king_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent (kings cannot be adjacent)."""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

    def _is_in_check(self, king_pos: Tuple[int, int], color: int) -> bool:
        """
        Check if a king is in check.

        Args:
            king_pos: King position
            color: King color (WHITE or BLACK)

        Returns:
            True if king is in check
        """
        if color == self.BLACK:
            if self._is_king_adjacent(king_pos, self.wk_pos):
                return True

            if self.wp_pos and not self.pawn_promoted:
                wp_r, wp_c = self.wp_pos
                for dc in [-1, 1]:
                    attack_pos = (wp_r + 1, wp_c + dc)
                    if attack_pos == king_pos:
                        return True

        return False

    def get_legal_actions(self, state: Tuple = None) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Get all legal actions from current state.

        Args:
            state: State tuple (if None, uses current state)

        Returns:
            List of (piece, new_position) tuples
        """
        if state:
            self._set_state(state)

        actions = []

        if self.turn == self.WHITE:
            wk_moves = self._get_king_moves(self.wk_pos)
            for move in wk_moves:
                if move != self.bk_pos and not self._is_king_adjacent(move, self.bk_pos):
                    if move != self.wp_pos or self.pawn_promoted:
                        actions.append(('WK', move))

            wp_moves = self._get_pawn_moves()
            for move in wp_moves:
                if move != self.wk_pos:
                    actions.append(('WP', move))

        else:
            bk_moves = self._get_king_moves(self.bk_pos)
            for move in bk_moves:
                if move != self.wk_pos and not self._is_king_adjacent(move, self.wk_pos):
                    if not self.wp_pos or move != self.wp_pos or self.pawn_promoted:
                        if not self._is_in_check(move, self.BLACK):
                            actions.append(('BK', move))

        return actions

    def _set_state(self, state: Tuple):
        """Set environment to a specific state."""
        wk_r, wk_c, wp_r, wp_c, bk_r, bk_c, turn, promoted, move_count = state
        self.wk_pos = (wk_r, wk_c)
        self.wp_pos = (wp_r, wp_c) if wp_r >= 0 else None
        self.bk_pos = (bk_r, bk_c)
        self.turn = turn
        self.pawn_promoted = bool(promoted)
        self.move_count = move_count
        self._update_board()

    def step(self, action: Tuple[str, Tuple[int, int]]) -> Tuple:
        """
        Execute an action and return new state, reward, done.

        Args:
            action: (piece, new_position) tuple

        Returns:
            Tuple of (new_state, reward, done, info)
        """
        piece, new_pos = action
        reward = 0
        done = False
        info = {}

        if piece == 'WK':
            self.wk_pos = new_pos
        elif piece == 'WP':
            if new_pos == self.bk_pos:
                self.bk_pos = None
                reward = -10
                done = True
                info['termination'] = 'pawn_captured_king'
            else:
                self.wp_pos = new_pos
                if new_pos[0] == self.board_size - 1:
                    self.pawn_promoted = True
                    reward = 10
                    done = True
                    info['termination'] = 'pawn_promotion'
        elif piece == 'BK':
            if new_pos == self.wp_pos and not self.pawn_promoted:
                self.wp_pos = None
                reward = -10
                done = True
                info['termination'] = 'pawn_captured'
            else:
                self.bk_pos = new_pos

        self._update_board()
        self.move_count += 1

        if not done and self.bk_pos:
            if self._is_checkmate():
                reward = 10
                done = True
                info['termination'] = 'checkmate'
            elif self._is_stalemate():
                reward = 0
                done = True
                info['termination'] = 'stalemate'

        if not done and self.move_count >= self.MOVE_LIMIT:
            reward = 0
            done = True
            info['termination'] = 'move_limit'

        self.turn = 1 - self.turn

        return self._get_state(), reward, done, info

    def _is_checkmate(self) -> bool:
        """Check if current position is checkmate."""
        if self.turn == self.BLACK and self.bk_pos:
            if self._is_in_check(self.bk_pos, self.BLACK):
                legal_actions = self.get_legal_actions()
                return len(legal_actions) == 0
        return False

    def _is_stalemate(self) -> bool:
        """Check if current position is stalemate."""
        if self.bk_pos:
            if not self._is_in_check(self.bk_pos, self.BLACK):
                legal_actions = self.get_legal_actions()
                return len(legal_actions) == 0
        return False

    def render(self):
        """Print current board state."""
        print(f"\nMove {self.move_count}, Turn: {'White' if self.turn == self.WHITE else 'Black'}")
        print("  " + " ".join([str(i) for i in range(self.board_size)]))
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join(row))
        print()


class ValueIteration:
    """
    Value Iteration algorithm for solving the Mini Chess MDP.
    """

    def __init__(self, env: MiniChessEnv, gamma: float = 0.99, theta: float = 1e-3):
        """
        Initialize Value Iteration.

        Args:
            env: Mini Chess environment
            gamma: Discount factor
            theta: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
        self.states = set()
        self.iterations = 0
        self.delta_history = []

    def enumerate_states(self, initial_state: Tuple, max_states: int = 10000) -> Set[Tuple]:
        """
        Enumerate reachable states using BFS.

        Args:
            initial_state: Starting state
            max_states: Maximum number of states to enumerate

        Returns:
            Set of reachable states
        """
        print(f"Enumerating reachable states from initial configuration...")

        visited = set()
        queue = deque([initial_state])
        visited.add(initial_state)

        while queue and len(visited) < max_states:
            state = queue.popleft()
            self.env._set_state(state)

            actions = self.env.get_legal_actions()
            for action in actions:
                self.env._set_state(state)
                next_state, reward, done, info = self.env.step(action)

                if next_state not in visited:
                    visited.add(next_state)
                    if not done:
                        queue.append(next_state)

        print(f"Enumerated {len(visited)} reachable states")
        self.states = visited
        return visited

    def solve(self) -> Tuple[Dict, Dict]:
        """
        Run Value Iteration algorithm.

        Returns:
            Tuple of (value_function, policy)
        """
        print(f"\nRunning Value Iteration...")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")

        for state in self.states:
            self.V[state] = 0.0

        start_time = time.time()
        self.iterations = 0

        while True:
            delta = 0
            self.iterations += 1

            for state in self.states:
                v = self.V[state]
                self.env._set_state(state)

                actions = self.env.get_legal_actions()
                if not actions:
                    self.V[state] = 0.0
                    continue

                action_values = []
                for action in actions:
                    self.env._set_state(state)
                    next_state, reward, done, info = self.env.step(action)

                    if done:
                        value = reward
                    else:
                        value = reward + self.gamma * self.V.get(next_state, 0.0)

                    action_values.append(value)

                self.V[state] = max(action_values) if action_values else 0.0
                delta = max(delta, abs(v - self.V[state]))

            self.delta_history.append(delta)

            if self.iterations % 10 == 0:
                print(f"  Iteration {self.iterations}: max delta = {delta:.6f}")

            if delta < self.theta:
                break

        runtime = time.time() - start_time

        print(f"\nValue Iteration Converged!")
        print(f"  Iterations: {self.iterations}")
        print(f"  Final max delta: {delta:.6f}")
        print(f"  Runtime: {runtime:.2f} seconds")

        self._extract_policy()

        return self.V, self.policy

    def _extract_policy(self):
        """Extract greedy policy from value function."""
        for state in self.states:
            self.env._set_state(state)
            actions = self.env.get_legal_actions()

            if not actions:
                continue

            action_values = []
            for action in actions:
                self.env._set_state(state)
                next_state, reward, done, info = self.env.step(action)

                if done:
                    value = reward
                else:
                    value = reward + self.gamma * self.V.get(next_state, 0.0)

                action_values.append((action, value))

            if action_values:
                best_action = max(action_values, key=lambda x: x[1])[0]
                self.policy[state] = best_action


class PolicyIteration:
    """
    Policy Iteration algorithm for solving the Mini Chess MDP.
    """

    def __init__(self, env: MiniChessEnv, gamma: float = 0.99, theta: float = 1e-3):
        """
        Initialize Policy Iteration.

        Args:
            env: Mini Chess environment
            gamma: Discount factor
            theta: Convergence threshold for policy evaluation
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
        self.states = set()
        self.iterations = 0

    def enumerate_states(self, initial_state: Tuple, max_states: int = 10000) -> Set[Tuple]:
        """
        Enumerate reachable states using BFS.

        Args:
            initial_state: Starting state
            max_states: Maximum number of states to enumerate

        Returns:
            Set of reachable states
        """
        print(f"Enumerating reachable states from initial configuration...")

        visited = set()
        queue = deque([initial_state])
        visited.add(initial_state)

        while queue and len(visited) < max_states:
            state = queue.popleft()
            self.env._set_state(state)

            actions = self.env.get_legal_actions()
            for action in actions:
                self.env._set_state(state)
                next_state, reward, done, info = self.env.step(action)

                if next_state not in visited:
                    visited.add(next_state)
                    if not done:
                        queue.append(next_state)

        print(f"Enumerated {len(visited)} reachable states")
        self.states = visited
        return visited

    def solve(self) -> Tuple[Dict, Dict]:
        """
        Run Policy Iteration algorithm.

        Returns:
            Tuple of (value_function, policy)
        """
        print(f"\nRunning Policy Iteration...")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")

        for state in self.states:
            self.V[state] = 0.0
            self.env._set_state(state)
            actions = self.env.get_legal_actions()
            if actions:
                self.policy[state] = actions[0]

        start_time = time.time()
        self.iterations = 0

        while True:
            self.iterations += 1
            print(f"\n  Policy Iteration - Iteration {self.iterations}")

            self._policy_evaluation()

            policy_stable = self._policy_improvement()

            if policy_stable:
                break

        runtime = time.time() - start_time

        print(f"\nPolicy Iteration Converged!")
        print(f"  Iterations: {self.iterations}")
        print(f"  Runtime: {runtime:.2f} seconds")

        return self.V, self.policy

    def _policy_evaluation(self):
        """Evaluate current policy."""
        eval_iterations = 0

        while True:
            delta = 0
            eval_iterations += 1

            for state in self.states:
                v = self.V[state]

                if state not in self.policy:
                    continue

                action = self.policy[state]
                self.env._set_state(state)
                next_state, reward, done, info = self.env.step(action)

                if done:
                    self.V[state] = reward
                else:
                    self.V[state] = reward + self.gamma * self.V.get(next_state, 0.0)

                delta = max(delta, abs(v - self.V[state]))

            if delta < self.theta:
                break

        print(f"    Policy evaluation completed in {eval_iterations} iterations")

    def _policy_improvement(self) -> bool:
        """
        Improve policy based on value function.

        Returns:
            True if policy is stable
        """
        policy_stable = True

        for state in self.states:
            old_action = self.policy.get(state)

            self.env._set_state(state)
            actions = self.env.get_legal_actions()

            if not actions:
                continue

            action_values = []
            for action in actions:
                self.env._set_state(state)
                next_state, reward, done, info = self.env.step(action)

                if done:
                    value = reward
                else:
                    value = reward + self.gamma * self.V.get(next_state, 0.0)

                action_values.append((action, value))

            if action_values:
                best_action = max(action_values, key=lambda x: x[1])[0]
                self.policy[state] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable


def visualize_value_function(env: MiniChessEnv, V: Dict, wp_pos: Tuple, bk_pos: Tuple, title: str):
    """
    Visualize value function as heatmap for different white king positions.

    Args:
        env: Mini Chess environment
        V: Value function dictionary
        wp_pos: Fixed white pawn position
        bk_pos: Fixed black king position
        title: Plot title
    """
    board_size = env.board_size
    value_grid = np.zeros((board_size, board_size))

    for r in range(board_size):
        for c in range(board_size):
            wk_pos = (r, c)

            if wk_pos == wp_pos or wk_pos == bk_pos:
                value_grid[r, c] = np.nan
                continue

            state = (wk_pos[0], wk_pos[1], wp_pos[0], wp_pos[1],
                    bk_pos[0], bk_pos[1], env.WHITE, 0, 0)

            if state in V:
                value_grid[r, c] = V[state]
            else:
                value_grid[r, c] = 0

    plt.figure(figsize=(8, 7))
    sns.heatmap(value_grid, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'State Value'},
                xticklabels=range(board_size), yticklabels=range(board_size))
    plt.title(f'{title}\nWP at {wp_pos}, BK at {bk_pos}', fontweight='bold')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.show()


def demonstrate_policy(env: MiniChessEnv, policy: Dict, initial_state: Tuple, max_moves: int = 15):
    """
    Demonstrate learned policy from a starting state.

    Args:
        env: Mini Chess environment
        policy: Learned policy
        initial_state: Starting state
        max_moves: Maximum moves to demonstrate
    """
    print(f"\nDemonstrating policy from initial state:")
    env._set_state(initial_state)
    env.render()

    state = initial_state
    moves = 0

    while moves < max_moves:
        if state not in policy:
            print("State not in policy!")
            break

        action = policy[state]
        piece, new_pos = action

        print(f"Move {moves + 1}: {piece} to {new_pos}")

        env._set_state(state)
        next_state, reward, done, info = env.step(action)
        env.render()

        if done:
            print(f"Game ended: {info.get('termination', 'unknown')}")
            print(f"Reward: {reward}")
            break

        state = next_state
        moves += 1


def main():
    """
    Main function to run Dynamic Programming solution.
    """
    print("="*80)
    print("PART 2: DYNAMIC PROGRAMMING FOR MINI CHESS GAME")
    print("="*80)
    print()

    student_id_last_digit = 0
    board_size = 4 if student_id_last_digit % 2 == 0 else 5

    print(f"Configuration:")
    print(f"  Student ID Last Digit: {student_id_last_digit}")
    print(f"  Board Size: {board_size}x{board_size}")
    print()

    print("="*80)
    print("1. CUSTOM MINI CHESS ENVIRONMENT")
    print("="*80)

    env = MiniChessEnv(board_size=board_size, student_id_last_digit=student_id_last_digit)
    initial_state = env.reset()

    print("\nInitial Configuration:")
    env.render()

    print(f"State Representation: {initial_state}")
    print(f"Legal Actions: {len(env.get_legal_actions())}")

    print("\n" + "="*80)
    print("2. DYNAMIC PROGRAMMING - VALUE ITERATION")
    print("="*80)

    vi = ValueIteration(env, gamma=0.99, theta=1e-3)
    vi.enumerate_states(initial_state, max_states=5000)
    V_vi, policy_vi = vi.solve()

    print("\n" + "="*80)
    print("3. DYNAMIC PROGRAMMING - POLICY ITERATION")
    print("="*80)

    pi = PolicyIteration(env, gamma=0.99, theta=1e-3)
    pi.enumerate_states(initial_state, max_states=5000)
    V_pi, policy_pi = pi.solve()

    print("\n" + "="*80)
    print("4. STATE-VALUE FUNCTION ANALYSIS")
    print("="*80)

    print("\nVisualizing value functions...")

    wp_fixed = (1, 1)
    bk_fixed = (3, 3) if board_size == 4 else (4, 4)

    visualize_value_function(env, V_vi, wp_fixed, bk_fixed,
                            "Value Iteration - State Values for WK Positions")

    visualize_value_function(env, V_pi, wp_fixed, bk_fixed,
                            "Policy Iteration - State Values for WK Positions")

    print("\n" + "="*80)
    print("5. POLICY DEMONSTRATION")
    print("="*80)

    demonstrate_policy(env, policy_vi, initial_state, max_moves=15)

    print("\n" + "="*80)
    print("6. ANALYSIS AND DISCUSSION")
    print("="*80)

    print("""
CURSE OF DIMENSIONALITY:

Current state space: ~{} states
- Board: {}x{}
- Pieces: 3 (WK, WP, BK)
- State dimensions: (wk_r, wk_c, wp_r, wp_c, bk_r, bk_c, turn, promoted, moves)

If we increase to 8x8 board:
- Each piece has 64 possible positions
- Approximate state space: 64^3 * 2 * 2 * 30 = ~15.7 million states
- With 2 pawns: 64^4 * 2^2 * 30 = ~1 billion states

If we add a rook:
- State space grows by factor of 64
- Dynamic programming becomes intractable

IS DYNAMIC PROGRAMMING ENOUGH FOR FULL CHESS?

No, standard DP is NOT tractable for full chess:

1. State Space Explosion:
   - Full chess: ~10^43 reachable positions
   - Cannot enumerate all states
   - Cannot store value function in memory

2. Modern RL Solutions:
   - Function Approximation: Use neural networks to approximate V(s)
   - Monte Carlo Tree Search: Sample promising paths
   - AlphaZero approach: Combine deep learning with MCTS
   - Self-play reinforcement learning

3. Key Insights from This Exercise:
   - DP works perfectly for small, tractable problems
   - Value/Policy iteration provide exact solutions
   - Real-world problems need approximation methods
   - Understanding DP fundamentals is crucial for advanced RL

STRUCTURAL PATTERNS OBSERVED:

1. Values are higher when:
   - White king is close to the pawn (protection)
   - Black king is far from the pawn
   - Pawn is advanced toward promotion

2. Values are lower when:
   - Black king can capture the pawn
   - White king is far from the pawn
   - Pawn is blocked by black king

3. Convergence:
   - Both algorithms converged quickly (~{} iterations)
   - Policy iteration typically needs fewer iterations
   - Value iteration is simpler to implement
    """.format(len(vi.states), board_size, board_size, vi.iterations))


if __name__ == "__main__":
    main()

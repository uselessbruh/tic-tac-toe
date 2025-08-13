import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Optional, Dict
from enum import Enum
import copy

class Player(Enum):
    EMPTY = 0
    AGENT = 1
    OPPONENT = 2

class GameResult(Enum):
    ONGOING = 0
    AGENT_WIN = 1
    OPPONENT_WIN = 2
    DRAW = 3

class SmartTicTacToeEnv(gym.Env):
    """
    Advanced Tic Tac Toe environment with intelligent opponent strategies
    """
    
    def __init__(self, opponent_strategy='smart', render_mode=None):
        """
        Initialize the environment
        
        Args:
            opponent_strategy: 'random', 'minimax', 'smart', 'adaptive'
            render_mode: None, 'human', 'rgb_array'
        """
        super().__init__()
        
        # Gym spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(18,), dtype=np.int8
        )  # 9 for board + 9 for valid moves mask
        
        # Game configuration
        self.opponent_strategy = opponent_strategy
        self.render_mode = render_mode
        
        # Winning combinations (rows, cols, diagonals)
        self.winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        
        # Strategic positions
        self.center = 4
        self.corners = [0, 2, 6, 8]
        self.edges = [1, 3, 5, 7]
        
        # Game state
        self.reset()
        
        # Performance tracking
        self.game_history = []
        self.opponent_adaptation_level = 0
        
    def reset(self, seed=None, options=None):
        """Reset the game to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = Player.AGENT
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.game_result = GameResult.ONGOING
        
        # Track game statistics
        self.agent_moves = []
        self.opponent_moves = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Position to place the agent's mark (0-8)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Validate agent move
        reward = 0
        if not self._is_valid_move(action):
            reward = -10  # Heavy penalty for invalid moves
            return self._get_observation(), reward, True, False, self._get_info()
        
        # Make agent move
        self._make_move(action, Player.AGENT)
        self.agent_moves.append(action)
        
        # Check if agent won
        if self._check_winner() == Player.AGENT:
            self.game_result = GameResult.AGENT_WIN
            reward = 10 + self._calculate_win_bonus()
            self.game_over = True
            return self._get_observation(), reward, True, False, self._get_info()
        
        # Check for draw
        if self._is_board_full():
            self.game_result = GameResult.DRAW
            reward = 1  # Small positive reward for achieving draw against smart opponent
            self.game_over = True
            return self._get_observation(), reward, True, False, self._get_info()
        
        # Opponent's turn
        opponent_action = self._get_opponent_move()
        if opponent_action is not None:
            self._make_move(opponent_action, Player.OPPONENT)
            self.opponent_moves.append(opponent_action)
            
            # Check if opponent won
            if self._check_winner() == Player.OPPONENT:
                self.game_result = GameResult.OPPONENT_WIN
                reward = -5 - self._calculate_loss_penalty()
                self.game_over = True
                return self._get_observation(), reward, True, False, self._get_info()
            
            # Check for draw after opponent move
            if self._is_board_full():
                self.game_result = GameResult.DRAW
                reward = 1
                self.game_over = True
                return self._get_observation(), reward, True, False, self._get_info()
        
        # Game continues - give small reward for strategic positions
        reward += self._calculate_positional_reward(action)
        
        return self._get_observation(), reward, False, False, self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including board state and valid moves"""
        valid_moves = np.array([1 if self._is_valid_move(i) else 0 for i in range(9)], dtype=np.int8)
        return np.concatenate([self.board, valid_moves])
    
    def _get_info(self) -> Dict:
        """Get additional information about the game state"""
        return {
            'move_count': self.move_count,
            'valid_moves': self._get_valid_moves(),
            'game_result': self.game_result.name,
            'board_state': self.board.copy(),
            'winner': self.winner.name if self.winner else None,
            'opponent_strategy': self.opponent_strategy,
            'agent_moves': self.agent_moves.copy(),
            'opponent_moves': self.opponent_moves.copy()
        }
    
    def _is_valid_move(self, action: int) -> bool:
        """Check if the move is valid"""
        return 0 <= action < 9 and self.board[action] == Player.EMPTY.value
    
    def _get_valid_moves(self) -> List[int]:
        """Get list of valid moves"""
        return [i for i in range(9) if self.board[i] == Player.EMPTY.value]
    
    def _make_move(self, action: int, player: Player):
        """Make a move on the board"""
        self.board[action] = player.value
        self.move_count += 1
    
    def _is_board_full(self) -> bool:
        """Check if the board is full"""
        return Player.EMPTY.value not in self.board
    
    def _check_winner(self) -> Optional[Player]:
        """Check if there's a winner"""
        for combo in self.winning_combos:
            if (self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != Player.EMPTY.value):
                winner = Player(self.board[combo[0]])
                self.winner = winner
                return winner
        return None
    
    def _get_opponent_move(self) -> Optional[int]:
        """Get opponent's move based on selected strategy"""
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return None
        
        if self.opponent_strategy == 'random':
            return self._random_opponent_move(valid_moves)
        elif self.opponent_strategy == 'minimax':
            return self._minimax_opponent_move()
        elif self.opponent_strategy == 'smart':
            return self._smart_opponent_move(valid_moves)
        elif self.opponent_strategy == 'adaptive':
            return self._adaptive_opponent_move(valid_moves)
        else:
            return self._smart_opponent_move(valid_moves)  # Default to smart
    
    def _random_opponent_move(self, valid_moves: List[int]) -> int:
        """Random opponent strategy"""
        return np.random.choice(valid_moves)
    
    def _minimax_opponent_move(self) -> int:
        """Minimax algorithm for optimal play"""
        def minimax(board, depth, is_maximizing, alpha=-np.inf, beta=np.inf):
            winner = self._check_winner_board(board)
            
            if winner == Player.OPPONENT:
                return 10 - depth
            elif winner == Player.AGENT:
                return depth - 10
            elif np.all(board != Player.EMPTY.value):
                return 0
            
            valid_positions = np.where(board == Player.EMPTY.value)[0]
            
            if is_maximizing:  # Opponent's turn
                max_eval = -np.inf
                for pos in valid_positions:
                    board[pos] = Player.OPPONENT.value
                    eval_score = minimax(board, depth + 1, False, alpha, beta)
                    board[pos] = Player.EMPTY.value
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                return max_eval
            else:  # Agent's turn
                min_eval = np.inf
                for pos in valid_positions:
                    board[pos] = Player.AGENT.value
                    eval_score = minimax(board, depth + 1, True, alpha, beta)
                    board[pos] = Player.EMPTY.value
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                return min_eval
        
        best_move = -1
        best_value = -np.inf
        board_copy = self.board.copy()
        
        for move in self._get_valid_moves():
            board_copy[move] = Player.OPPONENT.value
            move_value = minimax(board_copy, 0, False)
            board_copy[move] = Player.EMPTY.value
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move
    
    def _smart_opponent_move(self, valid_moves: List[int]) -> int:
        """
        Smart strategic opponent that follows advanced Tic Tac Toe strategy
        """
        # 1. Try to win immediately
        for move in valid_moves:
            if self._creates_winning_line(move, Player.OPPONENT):
                return move
        
        # 2. Block agent from winning
        for move in valid_moves:
            if self._creates_winning_line(move, Player.AGENT):
                return move
        
        # 3. Create fork (two ways to win)
        fork_moves = self._find_fork_moves(Player.OPPONENT)
        if fork_moves:
            return np.random.choice(fork_moves)
        
        # 4. Block agent's fork
        agent_fork_moves = self._find_fork_moves(Player.AGENT)
        if agent_fork_moves:
            # If agent has only one fork move, block it
            if len(agent_fork_moves) == 1:
                return agent_fork_moves[0]
            # If multiple fork moves, force agent to defend
            force_moves = self._find_force_moves(agent_fork_moves)
            if force_moves:
                return np.random.choice(force_moves)
        
        # 5. Take center if available (especially on first move)
        if self.center in valid_moves:
            if self.move_count <= 1:  # Prioritize center early
                return self.center
            # Take center if it provides strategic advantage
            if self._evaluate_position_value(self.center, Player.OPPONENT) > 3:
                return self.center
        
        # 6. Take opposite corner if agent took a corner
        opposite_corners = {0: 8, 2: 6, 6: 2, 8: 0}
        for corner in self.corners:
            if (self.board[corner] == Player.AGENT.value and 
                opposite_corners[corner] in valid_moves):
                return opposite_corners[corner]
        
        # 7. Take any available corner
        available_corners = [c for c in self.corners if c in valid_moves]
        if available_corners:
            # Prefer corners that create most opportunities
            corner_values = [(c, self._evaluate_position_value(c, Player.OPPONENT)) 
                           for c in available_corners]
            best_corner = max(corner_values, key=lambda x: x[1])[0]
            return best_corner
        
        # 8. Take best available edge
        available_edges = [e for e in self.edges if e in valid_moves]
        if available_edges:
            edge_values = [(e, self._evaluate_position_value(e, Player.OPPONENT)) 
                         for e in available_edges]
            best_edge = max(edge_values, key=lambda x: x[1])[0]
            return best_edge
        
        # 9. Fallback to random
        return np.random.choice(valid_moves)
    
    def _adaptive_opponent_move(self, valid_moves: List[int]) -> int:
        """
        Adaptive opponent that learns from agent's playing style
        """
        # Analyze agent's playing patterns
        self._update_adaptation_level()
        
        # Mix strategies based on adaptation level
        if self.opponent_adaptation_level < 3:
            # Start with smart strategy
            return self._smart_opponent_move(valid_moves)
        elif self.opponent_adaptation_level < 7:
            # Mix smart and minimax
            if np.random.random() < 0.7:
                return self._smart_opponent_move(valid_moves)
            else:
                return self._minimax_opponent_move()
        else:
            # Primarily use minimax with occasional surprises
            if np.random.random() < 0.9:
                return self._minimax_opponent_move()
            else:
                return self._smart_opponent_move(valid_moves)
    
    def _creates_winning_line(self, move: int, player: Player) -> bool:
        """Check if a move creates a winning line for the player"""
        board_copy = self.board.copy()
        board_copy[move] = player.value
        return self._check_winner_board(board_copy) == player
    
    def _check_winner_board(self, board: np.ndarray) -> Optional[Player]:
        """Check winner for a given board state"""
        for combo in self.winning_combos:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]] != Player.EMPTY.value):
                return Player(board[combo[0]])
        return None
    
    def _find_fork_moves(self, player: Player) -> List[int]:
        """Find moves that create forks (multiple winning opportunities)"""
        fork_moves = []
        valid_moves = self._get_valid_moves()
        
        for move in valid_moves:
            board_copy = self.board.copy()
            board_copy[move] = player.value
            
            # Count potential winning lines after this move
            winning_opportunities = 0
            for next_move in range(9):
                if board_copy[next_move] == Player.EMPTY.value:
                    board_copy[next_move] = player.value
                    if self._check_winner_board(board_copy) == player:
                        winning_opportunities += 1
                    board_copy[next_move] = Player.EMPTY.value
            
            if winning_opportunities >= 2:
                fork_moves.append(move)
        
        return fork_moves
    
    def _find_force_moves(self, opponent_fork_moves: List[int]) -> List[int]:
        """Find moves that force opponent to defend, preventing fork"""
        force_moves = []
        valid_moves = self._get_valid_moves()
        
        for move in valid_moves:
            if move not in opponent_fork_moves:
                # Check if this move creates a threat that opponent must respond to
                board_copy = self.board.copy()
                board_copy[move] = Player.OPPONENT.value
                
                # See if this creates an immediate winning threat
                for next_move in range(9):
                    if board_copy[next_move] == Player.EMPTY.value:
                        board_copy[next_move] = Player.OPPONENT.value
                        if self._check_winner_board(board_copy) == Player.OPPONENT:
                            force_moves.append(move)
                            break
                        board_copy[next_move] = Player.EMPTY.value
        
        return force_moves
    
    def _evaluate_position_value(self, position: int, player: Player) -> float:
        """Evaluate the strategic value of a position"""
        if self.board[position] != Player.EMPTY.value:
            return -1
        
        value = 0
        board_copy = self.board.copy()
        board_copy[position] = player.value
        
        # Count lines this position participates in
        lines_count = 0
        for combo in self.winning_combos:
            if position in combo:
                lines_count += 1
                # Check how many pieces player already has in this line
                player_pieces = sum(1 for pos in combo if self.board[pos] == player.value)
                empty_pieces = sum(1 for pos in combo if self.board[pos] == Player.EMPTY.value)
                opponent_pieces = sum(1 for pos in combo if self.board[pos] != player.value and self.board[pos] != Player.EMPTY.value)
                
                if opponent_pieces == 0:  # Line not blocked
                    value += (player_pieces + 1) ** 2
        
        # Bonus for strategic positions
        if position == self.center:
            value += 3
        elif position in self.corners:
            value += 2
        else:  # edges
            value += 1
        
        return value
    
    def _calculate_win_bonus(self) -> float:
        """Calculate bonus reward for winning quickly"""
        return max(0, 9 - self.move_count) * 0.5
    
    def _calculate_loss_penalty(self) -> float:
        """Calculate penalty for losing"""
        return min(self.move_count * 0.2, 2)
    
    def _calculate_positional_reward(self, action: int) -> float:
        """Calculate small rewards for good positional play"""
        reward = 0
        
        # Reward center play
        if action == self.center:
            reward += 0.3
        
        # Reward corner play
        elif action in self.corners:
            reward += 0.2
        
        # Small reward for participating in multiple lines
        lines_count = sum(1 for combo in self.winning_combos if action in combo)
        reward += lines_count * 0.1
        
        return reward
    
    def _update_adaptation_level(self):
        """Update opponent adaptation level based on game history"""
        if len(self.agent_moves) > 0:
            # Simple adaptation: increase level based on game patterns
            self.opponent_adaptation_level = min(10, len(self.game_history))
    
    def render(self, mode='human'):
        """Render the current game state"""
        if mode == 'human':
            symbols = {0: '·', 1: 'X', 2: 'O'}
            print("\nCurrent Board:")
            print("┌───┬───┬───┐")
            for i in range(0, 9, 3):
                row = " │ ".join(symbols[self.board[j]] for j in range(i, i + 3))
                print(f"│ {row} │")
                if i < 6:
                    print("├───┼───┼───┤")
            print("└───┴───┴───┘")
            
            if self.game_over:
                if self.winner == Player.AGENT:
                    print("🎉 Agent wins!")
                elif self.winner == Player.OPPONENT:
                    print("🤖 Opponent wins!")
                else:
                    print("🤝 It's a draw!")
            else:
                print(f"Move count: {self.move_count}")
                print(f"Valid moves: {self._get_valid_moves()}")
        
        elif mode == 'rgb_array':
            # For compatibility with rendering systems that expect RGB arrays
            # This would need to be implemented based on specific requirements
            pass
    
    def close(self):
        """Clean up resources"""
        pass

# Example usage and testing
if __name__ == "__main__":
    # Test the environment with different strategies
    strategies = ['random', 'smart', 'minimax', 'adaptive']
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing with {strategy} opponent")
        print('='*50)
        
        env = SmartTicTacToeEnv(opponent_strategy=strategy)
        obs, info = env.reset()
        
        # Simple random agent for testing
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            # Random agent action
            valid_moves = info['valid_moves']
            if not valid_moves:
                break
                
            action = np.random.choice(valid_moves)
            obs, reward, done, truncated, info = env.step(action)
            
            env.render()
            print(f"Agent action: {action}, Reward: {reward:.2f}")
            
            step_count += 1
            
            if done:
                print(f"\nGame finished! Result: {info['game_result']}")
                print(f"Total steps: {step_count}")
                break
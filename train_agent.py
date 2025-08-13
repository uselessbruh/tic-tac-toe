from stable_baselines3 import DQN
from tictactoe_env import SmartTicTacToeEnv  # Update this import to match your file name
from stable_baselines3.common.env_checker import check_env

# Create environment
env = SmartTicTacToeEnv(opponent_strategy='smart')

# Check environment
check_env(env, warn=True)

# Create and train model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save model
model.save("tictactoe_agent")
print("Training complete. Model saved.")
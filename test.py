from stable_baselines3 import DQN
from tictactoe_env import SmartTicTacToeEnv

model = DQN.load("tictactoe_agent")
env = SmartTicTacToeEnv()

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()

print("Reward:", reward)

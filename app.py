from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import DQN
from tictactoe_env import SmartTicTacToeEnv
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and env only once
model = DQN.load("tictactoe_agent")
env = SmartTicTacToeEnv()

@app.route("/move", methods=["POST"])
def get_ai_move():
    data = request.get_json()
    board = data.get("board", [])

    if not board or len(board) != 9:
        return jsonify({"error": "Invalid board"}), 400

    env.board = np.array(board, dtype=np.int8)
    obs = env.board
    action, _ = model.predict(obs, deterministic=True)

    return jsonify({"move": int(action)})

# 🚨 THIS IS REQUIRED TO KEEP THE SERVER RUNNING
if __name__ == "__main__":
    app.run(debug=True)

from game import Game
from agent import RandomAgent, AlternateAgent, BasePlanningAgent  # or your custom agents
from expert_params import expert_params

import numpy as np

def benchmark(agent, num_games=1, game_target=25, bot_memory_reset=True):
    results = []

    for _ in range(num_games):
        g = Game(game_target, expert_params, agent=agent, bot_memory_reset=bot_memory_reset, benchmark_mode=True)
        winner, turns = g.play_game()

        # Convert win = +turns, loss = −turns for metric
        score = turns if winner == "user" else -turns
        results.append(score)

    results = np.array(results)
    wins = np.sum(results > 0)
    losses = np.sum(results < 0)

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Games played: {num_games}")
    print(f"Wins: {wins} ({wins/num_games:.1%})")
    print(f"Losses: {losses} ({losses/num_games:.1%})")
    print(f"Average score (± turns): {results.mean():.2f}")
    print(f"Std dev: {results.std():.2f}")

    return results


if __name__ == "__main__":
    # agent = RandomAgent()  # replace with other agents
    # agent = AlternateAgent()
    agent = BasePlanningAgent(verbose=True)
    benchmark(agent, num_games=10, game_target=25, bot_memory_reset=True)

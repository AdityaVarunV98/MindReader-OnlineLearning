# from game import Game
# from agent import RandomAgent, AlternateAgent, BasePlanningAgent, SimpleMCPlanningAgent  # or your custom agents
# from expert_params import expert_params

# import numpy as np

# def benchmark(agent, num_games=1, game_target=25, bot_memory_reset=True):
#     results = []

#     for _ in range(num_games):
#         g = Game(game_target, expert_params, agent=agent, bot_memory_reset=bot_memory_reset, benchmark_mode=True)
#         winner, turns = g.play_game()

#         # Convert win = +turns, loss = −turns for metric
#         score = turns if winner == "user" else -turns
#         results.append(score)
#         print("Game number: ", _)

#     results = np.array(results)
#     wins = np.sum(results > 0)
#     losses = np.sum(results < 0)

#     print("\n===== BENCHMARK RESULTS =====")
#     print(f"Games played: {num_games}")
#     print(f"Wins: {wins} ({wins/num_games:.1%})")
#     print(f"Losses: {losses} ({losses/num_games:.1%})")
#     print(f"Average score (± turns): {results.mean():.2f}")
#     print(f"Std dev: {results.std():.2f}")

#     return results


# if __name__ == "__main__":
#     agent = SimpleMCPlanningAgent()

#     benchmark(agent, num_games=100, game_target=25, bot_memory_reset=True)


from game import Game
from expert_params import expert_params
from MRPythonImplementation.BaselineImplementation.mcts_agent import MCTSAgent  # ← New import

import numpy as np
import argparse


def benchmark(agent, num_games=1, game_target=25, bot_memory_reset=True):
    results = []

    for i in range(num_games):
        g = Game(game_target, expert_params, agent=agent, bot_memory_reset=bot_memory_reset, benchmark_mode=True)
        winner, turns = g.play_game()

        # Convert win = +turns, loss = −turns for metric 
        # CHANGE THIS!!
        score = 2*game_target - turns if winner == "user" else -1*(2*game_target - turns) - 1
        results.append(score)

        print(f"Game {i+1}/{num_games} finished — Winner: {winner.upper()} in {turns} turns.")

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
    parser = argparse.ArgumentParser(description="Benchmark MCTSAgent vs Bot")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--target", type=int, default=25, help="Target score to win")
    parser.add_argument("--sims", type=int, default=500, help="Number of MCTS simulations per move")
    parser.add_argument("--c", type=float, default=1.4, help="UCT exploration constant")
    parser.add_argument("--rollout", type=int, default=1000, help="Maximum rollout steps per simulation")
    parser.add_argument("--no-reset", action="store_true", help="Keep bot memory across games")
    parser.add_argument("--verbose", action="store_true", help="Print detailed MCTS stats")

    args = parser.parse_args()

    agent = MCTSAgent(
        n_simulations=args.sims,
        c_puct=args.c,
        rollout_limit=args.rollout,
        verbose=args.verbose
    )

    benchmark(
        agent,
        num_games=args.games,
        game_target=args.target,
        bot_memory_reset=not args.no_reset
    )

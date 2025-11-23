from game import Game
from expert_params import expert_params
from az_mcts_learn import AlphaZeroMCTSAgent, StateEncoder, Net
import torch
import numpy as np
import argparse


def benchmark(agent, num_games=1, game_target=25, bot_memory_reset=True):
    results = []

    for i in range(num_games):
        g = Game(game_target, expert_params, agent=agent, bot_memory_reset=bot_memory_reset, benchmark_mode=True)
        winner, turns = g.play_game()

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
    parser = argparse.ArgumentParser(description="Benchmark AlphaZero Agent vs Bot")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--target", type=int, default=25, help="Target score to win")
    parser.add_argument("--sims", type=int, default=200, help="Number of MCTS simulations per move")
    parser.add_argument("--c", type=float, default=1.25, help="PUCT exploration constant")
    parser.add_argument("--no-reset", action="store_true", help="Keep bot memory across games")
    parser.add_argument("--verbose", action="store_true", help="Print detailed MCTS stats")
    parser.add_argument("--model", type=str, default="az_agent.pth", help="Path to trained AlphaZero model")

    args = parser.parse_args()

    # === Device setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Initialize encoder and network ===
    encoder = StateEncoder(max_history=40)
    input_dim = 262  # fixed based on your model architecture

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Rebuild and load AlphaZero agent ===
    try:
        # Recommended if your AlphaZeroMCTSAgent defines its own `load()` method
        agent = AlphaZeroMCTSAgent.load(args.model, encoder, input_dim=input_dim)
        if args.verbose:
            print(f"[Load] Loaded full AlphaZero agent from {args.model}")
    except AttributeError:
        # Fallback: manually rebuild from Net and encoder
        net = Net(input_dim=input_dim, hidden_dim=128).to(device)
        state_dict = torch.load(args.model, map_location=device)
        if "state_dict" in state_dict:  # handle case where dict has nested structure
            net.load_state_dict(state_dict["state_dict"])
        else:
            net.load_state_dict(state_dict)
        net.eval()

        agent = AlphaZeroMCTSAgent(
            net=net,
            encoder=encoder,
            n_simulations=args.sims,
            c_puct=args.c,
            verbose=args.verbose
        )
        if args.verbose:
            print(f"[Load] Loaded network weights from {args.model}")

    agent.use_mcts = False

    # === Run benchmark ===
    benchmark(
        agent,
        num_games=args.games,
        game_target=args.target,
        bot_memory_reset=not args.no_reset
    )

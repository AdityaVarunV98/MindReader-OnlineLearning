import numpy as np
import copy
from collections import defaultdict
from types import SimpleNamespace


class BaseAgent:
    """
    Base class for agents. Any agent must implement:
        act(game_state) -> returns -1 (left) or +1 (right)
    """
    def __init__(self, name="BaseAgent"):
        self.name = name

    def act(self, game_state):
        raise NotImplementedError("Agent must implement act().")


# Chooses based on simulations of one-step lookahead
class BasePlanningAgent:
    def __init__(self, n_simulations=10, epsilon=0.1, reward_weights=None, verbose=False):
        self.n_simulations = n_simulations
        self.epsilon = epsilon
        self.verbose = verbose

        self.reward_weights = reward_weights or {
            "bias_factor": 1.0,
            "step": -0.01,
            "win": 10.0,
            "loss": -10.0
        }

        # Debug tracking for analysis
        self.turn_stats = {"turn": [], "q_left": [], "q_right": [], "chosen_action": []}

    def act(self, game_state):
        """
        Perform lookahead rollouts from current state and pick best move.
        game_state is a dict: {"bot": bot_copy, "game": game_copy}
        """
        bot = game_state["bot"]
        game = game_state["game"]

        possible_actions = [-1, 1]
        values = {}

        # Simulate outcomes for each possible action
        for a in possible_actions:
            values[a] = self.simulate_action(bot, game, a)

        # Debug logging
        if self.verbose:
            print(f"\n[Turn {game.turn_number}] Q-value estimates:")
            for a, val in values.items():
                print(f"  Action {a:+d} → {val:.4f}")

        # Store Q-values for plotting later
        self.turn_stats["turn"].append(game.turn_number)
        self.turn_stats["q_left"].append(values[-1])
        self.turn_stats["q_right"].append(values[1])

        # Choose ε-greedily
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(possible_actions)
            if self.verbose:
                print(f"  [ε-Greedy] Randomly chose action {chosen:+d}")
        else:
            chosen = max(values, key=values.get)
            if self.verbose:
                print(f"  [Greedy] Chose best action {chosen:+d}")

        self.turn_stats["chosen_action"].append(chosen)
        return chosen

    def simulate_action(self, bot, game, action):
        """
        Perform multiple rollout simulations starting from (game, action).
        """
        total_return = 0
        for sim in range(self.n_simulations):
            reward = self.run_simulation(bot, game, action)
            total_return += reward

            if self.verbose and sim == 0:
                print(f"    Simulation {sim + 1}: Reward = {reward:.4f}")

        return total_return / self.n_simulations

    def run_simulation(self, bot, game, action):
        """
        Run one simulated rollout starting from the given game state and chosen action.
        """
        # Deepcopy so each simulation is independent
        sim_game = copy.deepcopy(game)
        sim_bot = copy.deepcopy(bot)

        # Step 1: Apply the agent's chosen move
        sim_game.user_strokes.append(action)

        # Step 2: Bot responds using the full copied game
        _, bot_move = sim_bot.bot_play(sim_game)
        sim_game.bot_strokes.append(bot_move)

        # Step 3: Compute immediate reward
        reward = self.compute_reward(sim_game.user_strokes, sim_game.bot_strokes,
                                     sim_game.user_grade, sim_game.bot_grade,
                                     sim_game.game_target)

        # if self.verbose:
        #     print(f"      Turn {sim_game.turn_number}: Agent={action:+d}, Bot={bot_move:+d}, Reward={reward:.3f}")

        return reward

    def compute_reward(self, user_strokes, bot_strokes, user_grade, bot_grade, target):
        """
        Compute shaped reward for a single step.
        """
        # Update win/loss outcome
        if bot_strokes[-1] == user_strokes[-1]:
            bot_grade += 1
            win_loss = -1
        else:
            user_grade += 1
            win_loss = 1

        # Reward shaping
        bias_term = self.reward_weights["bias_factor"] * win_loss
        step_penalty = self.reward_weights["step"]
        terminal_reward = 0
        if user_grade >= target:
            terminal_reward = self.reward_weights["win"]
        elif bot_grade >= target:
            terminal_reward = self.reward_weights["loss"]

        return bias_term + step_penalty + terminal_reward


class RandomAgent(BaseAgent):
    """Plays randomly with equal probability"""
    def __init__(self):
        super().__init__(name="RandomAgent")

    def act(self, game_state):
        return 2 * np.random.binomial(1, 0.5) - 1


class AlternateAgent(BaseAgent):
    """Plays randomly with equal probability"""
    def __init__(self):
        super().__init__(name="RandomAgent")

    def act(self, game_state):
        if game_state["turn"] % 2 == 0:
            return -1
        else:
            return 1


class HumanAgent(BaseAgent):
    """Human user input: a=left, d=right, q=quit"""
    def __init__(self):
        super().__init__(name="HumanAgent")

    def act(self, game_state):
        while True:
            stroke = input("Your move [a/d/q]: ").strip().lower()
            if stroke == "d":
                return 1
            elif stroke == "a":
                return -1
            elif stroke == "q":
                return None
            else:
                print("Invalid input.")

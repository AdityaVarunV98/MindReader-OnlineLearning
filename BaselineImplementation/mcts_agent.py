import copy
import math
import numpy as np
import random
from types import SimpleNamespace

class MCTSNode:
    def __init__(self, parent=None, prior_action=None):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior_action = prior_action  # action that led to this node (for debugging)

    def q(self):
        return self.value_sum

    def value(self):
        # mean value
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_leaf(self):
        return len(self.children) == 0

class MCTSAgent:
    """
    MCTS agent for the MindReader environment.
    act(game_state) -> returns -1 or +1

    Parameters:
        n_simulations: total MCTS iterations per decision
        c_puct: exploration constant in UCT
        rollout_limit: maximum steps allowed in a rollout (safety)
        verbose: print debug info
    """

    def __init__(self, n_simulations=500, c_puct=1.4, rollout_limit=1000, verbose=False):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.rollout_limit = rollout_limit
        self.verbose = verbose

        # statistics for optional debugging
        self.root_visits_history = []

    def act(self, game_state):
        """
        game_state: {"bot": bot_copy, "game": game_copy}
        Note: Game.get_user_move passes deepcopied bot and game into agent.act,
        but we will deepcopy inside simulations too to be safe.
        """
        root_bot = game_state["bot"]
        root_game = game_state["game"]

        # Root corresponds to the state before agent's move (bot has already moved this turn)
        root = MCTSNode(parent=None)

        # available actions for agent
        actions = [-1, 1]

        # initialize children (lazy expansion allowed; here we pre-create children)
        for a in actions:
            root.children[a] = MCTSNode(parent=root, prior_action=a)

        for i in range(self.n_simulations):
            node = root
            sim_bot = copy.deepcopy(root_bot)
            sim_game = copy.deepcopy(root_game)

            # ---------- SELECTION ----------
            # traverse tree: choose child by UCT until we hit a leaf node
            path = [node]
            while not node.is_leaf():
                # choose child with highest UCT value
                best_action, best_child = None, None
                best_score = -float("inf")
                for act, child in node.children.items():
                    if child.visits == 0:
                        # prioritize unvisited nodes
                        score = float("inf")
                    else:
                        # exploitation term: mean value (we interpret agent's perspective: user win = +1)
                        exploit = child.value()
                        explore = self.c_puct * math.sqrt(math.log(node.visits + 1) / child.visits)
                        score = exploit + explore
                    if score > best_score:
                        best_score = score
                        best_action = act
                        best_child = child

                # apply selected action (best_action) to simulated environment
                # Agent plays
                sim_game.user_strokes.append(best_action)
                # Bot responds
                _, bot_move = sim_bot.bot_play(sim_game)
                sim_game.bot_strokes.append(bot_move)
                # update scores/state after both have played the turn
                sim_game.update_status()

                node = best_child
                path.append(node)

                # If terminal reached, stop selection
                if sim_game.user_grade >= sim_game.game_target or sim_game.bot_grade >= sim_game.game_target:
                    break

            # ---------- EXPANSION ----------
            # If node is leaf and non-terminal, expand by adding its children (all actions)
            terminal = (sim_game.user_grade >= sim_game.game_target or sim_game.bot_grade >= sim_game.game_target)
            if (not terminal) and node.is_leaf():
                for a in actions:
                    if a not in node.children:
                        node.children[a] = MCTSNode(parent=node, prior_action=a)

                # After expansion, pick one child uniformly for rollout start
                pick_action = random.choice(actions)
                node = node.children[pick_action]

                # Apply that chosen action to sim state
                sim_game.user_strokes.append(pick_action)
                _, bot_move = sim_bot.bot_play(sim_game)
                sim_game.bot_strokes.append(bot_move)
                sim_game.update_status()

                path.append(node)

            # ---------- SIMULATION / ROLLOUT ----------
            # Run a random rollout from the current sim_game until terminal or rollout_limit
            rollout_steps = 0
            while sim_game.user_grade < sim_game.game_target and sim_game.bot_grade < sim_game.game_target and rollout_steps < self.rollout_limit:
                # agent (during rollout) plays randomly
                rollout_action = random.choice(actions)
                sim_game.user_strokes.append(rollout_action)

                # bot responds according to its model
                _, bot_move = sim_bot.bot_play(sim_game)
                sim_game.bot_strokes.append(bot_move)

                sim_game.update_status()
                rollout_steps += 1

            # ---------- BACKPROPAGATION ----------
            # Assign outcome: +1 if user (agent) won, -1 if bot won. If cutoff, use 0.
            if sim_game.user_grade >= sim_game.game_target and sim_game.user_grade > sim_game.bot_grade:
                outcome = 1.0
            elif sim_game.bot_grade >= sim_game.game_target and sim_game.bot_grade > sim_game.user_grade:
                outcome = -1.0
            else:
                # rollout limit reached or tie-ish; we can treat as 0 (neutral)
                outcome = 0.0

            # propagate outcome up the path
            for visited_node in path:
                visited_node.visits += 1
                visited_node.value_sum += outcome

        # Choose final action: child with highest visit count (robust) or highest value
        best_action = None
        best_visits = -1
        for act, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = act

        # optional debugging
        if self.verbose:
            print("MCTS root children stats:")
            for a, child in root.children.items():
                print(f"  action {a:+d} → visits={child.visits}, mean_value={child.value():.3f}")

        return best_action

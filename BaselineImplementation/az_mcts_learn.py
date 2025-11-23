import copy
import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Neural net: policy + value (BIGGER)
# -------------------------
class Net(nn.Module):
    """
    MLP that outputs:
      - policy logits for two actions (order: [-1, +1])
      - scalar value in [-1, 1] (tanh)
    Input: feature vector produced by StateEncoder

    Changed:
      - Increased hidden_dim
      - Deeper trunk for more capacity
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2)  # two actions
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.policy_head(h)
        # keep value bounded in [-1, 1]
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return logits, value

# -------------------------
# Encoder: pack game+bot state into fixed-length vector
# -------------------------
class StateEncoder:
    """
    Convert game+bot into fixed-length vector input for NN.

    Strategy:
     - Extract recent history up to `max_history` moves:
         user_strokes, user_strokes_same_diff, user_win_loss,
         bot_strokes, bot_strokes_same_diff, bot_win_loss
     - Flatten and pad/truncate to fixed length.
     - Include scalar features: turn_number, user_grade, bot_grade, game_target.
     - Include richer bot.current_bot_status["experts"] summary:
         * mean across experts (per group)
         * std across experts (per group)
         * optionally: mean over recent time window
    """
    def __init__(self, max_history=40, expert_history_window=5):
        self.max_history = max_history
        self.expert_history_window = expert_history_window

    def pad_truncate(self, arr, pad_value=0.0):
        # latest turns at the end of arr; we want fixed length with zeros at front if short
        arr = list(arr[-self.max_history:])  # keep last max_history
        if len(arr) < self.max_history:
            return [pad_value] * (self.max_history - len(arr)) + arr
        return arr

    def encode(self, bot, game):
        # Basic sequences
        us = self.pad_truncate(game.user_strokes, pad_value=0.0)
        us_sd = self.pad_truncate(game.user_strokes_same_diff, pad_value=0.0)
        uwl = self.pad_truncate(game.user_win_loss, pad_value=0.0)

        bs = self.pad_truncate(game.bot_strokes, pad_value=0.0)
        bs_sd = self.pad_truncate(game.bot_strokes_same_diff, pad_value=0.0)
        bwl = self.pad_truncate(game.bot_win_loss, pad_value=0.0)

        # scalar features
        turn = float(game.turn_number) / max(game.game_target, 1)  # normalize roughly
        ug = float(game.user_grade) / max(game.game_target, 1)
        bg = float(game.bot_grade) / max(game.game_target, 1)
        target = float(game.game_target)

        # Bot experts summary: richer statistics
        experts = bot.current_bot_status.get("experts", None)
        if experts is not None and experts.size > 0:
            # expected shape: (n_groups=6, n_experts_per_group=M, history_len)
            if experts.ndim == 3:
                # last time-slice
                last_slice = (
                    experts[:, :, -1]
                    if experts.shape[2] > 0
                    else experts[:, :, :1].mean(axis=2)
                )  # shape (6, M)

                # recent window mean (over last self.expert_history_window time-steps)
                if experts.shape[2] >= self.expert_history_window:
                    window_slice = experts[:, :, -self.expert_history_window :]
                else:
                    window_slice = experts

                # aggregate stats over experts/time
                group_mean_last = np.nanmean(last_slice, axis=1)        # (6,)
                group_std_last = np.nanstd(last_slice, axis=1)          # (6,)
                group_mean_window = np.nanmean(window_slice, axis=(1, 2))  # (6,)

                expert_features = np.concatenate(
                    [group_mean_last, group_std_last, group_mean_window], axis=0
                )  # length = 18
            else:
                # fallback if unexpected shape
                expert_features = np.zeros(18, dtype=np.float32)
        else:
            expert_features = np.zeros(18, dtype=np.float32)

        feat = (
            list(us) + list(us_sd) + list(uwl) +
            list(bs) + list(bs_sd) + list(bwl) +
            [turn, ug, bg, target] +
            list(expert_features)
        )
        return np.array(feat, dtype=np.float32)

# -------------------------
# MCTS node (modified to store priors)
# -------------------------
class MCTSNode:
    def __init__(self, parent=None, prior=0.0, action=None):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior   # prior probability from policy network
        self.action = action

    def q(self):
        return self.value_sum

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_leaf(self):
        return len(self.children) == 0



# ===== BENCHMARK RESULTS =====
# Games played: 100
# Wins: 86 (86.0%)
# Losses: 14 (14.0%)
# Average score (± turns): 7.36
# Std dev: 7.92
# (az_agent_2.pth)
# Game target of 50

# -------------------------
# AlphaZero-like MCTS Agent
# -------------------------
class AlphaZeroMCTSAgent:
    """
    MCTS agent using a policy+value network.
    """
    def __init__(self, net, encoder: StateEncoder,
                 n_simulations=400,  # increased simulations
                 c_puct=1.25,
                 device=None, use_mcts=True, verbose=False):
        self.net = net
        self.encoder = encoder
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.verbose = verbose
        self.use_mcts = use_mcts

        # training buffer (list of training examples from self-play)
        # Each example: (state_vector, policy_target (2,), value_target scalar)
        self.train_buffer = []

    def _policy_only_action(self, game_state, temperature=0.0):
        """
        Use only the policy network (no MCTS), for fast test-time decisions.
        """
        root_bot = game_state["bot"]
        root_game = game_state["game"]

        state_vec = self.encoder.encode(root_bot, root_game)
        with torch.no_grad():
            logits, _ = self.net(
                torch.from_numpy(state_vec).to(self.device).unsqueeze(0)
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()  # shape (2,)

        actions = [-1, 1]

        if temperature == 0:
            # purely greedy: pick the argmax of probs
            best_idx = int(np.argmax(probs))
            chosen_action = actions[best_idx]
        else:
            # soft sampling from policy distribution
            probs = probs / (np.sum(probs) + 1e-12)
            chosen_action = np.random.choice(actions, p=probs)

        return chosen_action

    
    # -----------------------------------------------------------------------------
    # MCTS (core)
    # -----------------------------------------------------------------------------
    def act(self, game_state, temperature=1.0, return_search_stats=False):
        """
        game_state: {"bot": bot_copy, "game": game_copy}
        Returns chosen action (-1 or +1). Optionally returns search stats (visit counts).
        """

        # ---------- TEST MODE: policy-only, no simulations ----------
        if not self.use_mcts:
            action = self._policy_only_action(game_state, temperature=0.0 if temperature is None else temperature)
            if return_search_stats:
                # In test mode we don't have tree statistics; we can still return policy probs if needed.
                root_bot = game_state["bot"]
                root_game = game_state["game"]
                state_vec = self.encoder.encode(root_bot, root_game)
                with torch.no_grad():
                    logits, _ = self.net(
                        torch.from_numpy(state_vec).to(self.device).unsqueeze(0)
                    )
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                search_info = {
                    "state_vec": state_vec,
                    "policy_target": probs,
                    "root": None
                }
                return action, search_info
            return action

        root_bot = game_state["bot"]
        root_game = game_state["game"]

        # initialize root node and get priors for root state
        root = MCTSNode(parent=None, prior=0.0, action=None)
        state_vec = self.encoder.encode(root_bot, root_game)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(state_vec).to(self.device).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()  # shape (2,)

        # Map actions -> indices: idx0 -> action -1, idx1 -> action +1
        actions = [-1, 1]
        for idx, a in enumerate(actions):
            root.children[a] = MCTSNode(parent=root, prior=float(probs[idx]), action=a)

        # run simulations (fresh tree each call)
        for _ in range(self.n_simulations):
            node = root
            sim_bot = copy.deepcopy(root_bot)
            sim_game = copy.deepcopy(root_game)
            path = [node]

            # SELECTION
            while not node.is_leaf():
                # select child maximizing PUCT
                best_score = -float("inf")
                best_child = None
                best_action = None
                for a, child in node.children.items():
                    if child.visits == 0:
                        q = 0.0
                    else:
                        q = child.value()
                    # PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child)
                    u = self.c_puct * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_child = child
                        best_action = a

                # apply action and bot response
                sim_game.user_strokes.append(best_action)
                _, bot_move = sim_bot.bot_play(sim_game)
                sim_game.bot_strokes.append(bot_move)
                sim_game.update_status()

                node = best_child
                path.append(node)

                # stop if terminal
                if sim_game.user_grade >= sim_game.game_target or sim_game.bot_grade >= sim_game.game_target:
                    break

            # EXPANSION & EVALUATION
            terminal = (sim_game.user_grade >= sim_game.game_target or sim_game.bot_grade >= sim_game.game_target)
            if not terminal and node.is_leaf():
                # compute priors and value for this leaf
                leaf_state_vec = self.encoder.encode(sim_bot, sim_game)
                with torch.no_grad():
                    logits, value = self.net(torch.from_numpy(leaf_state_vec).to(self.device).unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                    value = float(value.cpu().numpy().squeeze())

                # create children with priors
                for idx, a in enumerate(actions):
                    if a not in node.children:
                        node.children[a] = MCTSNode(parent=node, prior=float(probs[idx]), action=a)

                # no rollout: we will backpropagate the network value from this leaf
                eval_value = value
            else:
                # terminal node: assign deterministic outcome
                if sim_game.user_grade >= sim_game.game_target and sim_game.user_grade > sim_game.bot_grade:
                    eval_value = 1.0
                elif sim_game.bot_grade >= sim_game.game_target and sim_game.bot_grade > sim_game.user_grade:
                    eval_value = -1.0
                else:
                    eval_value = 0.0

            # BACKUP: propagate eval_value up the path
            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += eval_value

        # After simulations: compute visit counts and choose best action
        visits = np.array([root.children[a].visits for a in actions], dtype=np.float32)
        # temperature handling: higher temperature -> more exploration in action selection
        if temperature == 0:
            # pick argmax visits
            best_idx = int(np.argmax(visits))
            chosen_action = actions[best_idx]
        else:
            probs_vis = visits ** (1.0 / temperature)
            probs_vis = probs_vis / (np.sum(probs_vis) + 1e-12)
            chosen_action = np.random.choice(actions, p=probs_vis)

        # build training example: state_vec -> policy_target (normalized visit counts)
        policy_target = visits / (np.sum(visits) + 1e-12)
        search_info = {
            "state_vec": state_vec,
            "policy_target": policy_target,
            "root": root
        }
        if return_search_stats:
            return chosen_action, search_info
        return chosen_action

    # -----------------------------------------------------------------------------
    # Self-play and training orchestration
    # -----------------------------------------------------------------------------
    def play_game_and_collect(self, game):
        """
        Play one full game where `game.agent` is this agent (set externally), collecting MCTS data.
        Returns training examples: list of (state_vec, policy_target, final_value) where
        final_value is +1/-1 from user's perspective.
        """
        game.agent = self

        sim_game = copy.deepcopy(game)
        sim_bot = copy.deepcopy(game.bot)

        examples = []
        while sim_game.user_grade < sim_game.game_target and sim_game.bot_grade < sim_game.game_target:
            # Bot plays first
            sim_bot, bot_move = sim_bot.bot_play(sim_game)
            sim_game.bot_strokes.append(bot_move)

            # Agent move: call act and get search info
            chosen_action, search_info = self.act(
                {"bot": copy.deepcopy(sim_bot), "game": copy.deepcopy(sim_game)},
                return_search_stats=True
            )
            sim_game.user_strokes.append(chosen_action)

            # update status
            sim_game.update_status()

            # save example; value target to be filled after game ends
            examples.append({
                "state_vec": search_info["state_vec"],
                "policy_target": search_info["policy_target"]
            })

        # compute final outcome from user's perspective
        if sim_game.user_grade > sim_game.bot_grade:
            final_value = 1.0
        elif sim_game.bot_grade > sim_game.user_grade:
            final_value = -1.0
        else:
            final_value = 0.0

        # produce training tuples
        training_examples = []
        for ex in examples:
            training_examples.append((ex["state_vec"], ex["policy_target"], final_value))

        return training_examples, ("user" if final_value == 1.0 else "bot", sim_game.turn_number - 1)

    def add_to_buffer(self, examples, max_buffer_size=50000):
        # extend; optionally cap replay buffer size
        self.train_buffer.extend(examples)
        if len(self.train_buffer) > max_buffer_size:
            # keep most recent max_buffer_size examples
            self.train_buffer = self.train_buffer[-max_buffer_size:]

    def train_from_buffer(self, batch_size=128, epochs=3, lr=1e-3):
        if len(self.train_buffer) == 0:
            return

        # simple dataset creation (numpy -> tensors)
        X = np.stack([e[0] for e in self.train_buffer], axis=0).astype(np.float32)
        P = np.stack([e[1] for e in self.train_buffer], axis=0).astype(np.float32)  # (N,2)
        V = np.array([e[2] for e in self.train_buffer], dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X), torch.from_numpy(P), torch.from_numpy(V)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.net.parameters(), lr=lr)
        mse = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            for xb, pb, vb in loader:
                xb = xb.to(self.device)
                pb = pb.to(self.device)
                vb = vb.to(self.device)

                logits, vals = self.net(xb)
                logp = torch.log_softmax(logits, dim=-1)
                policy_loss = -torch.sum(pb * logp, dim=-1).mean()

                value_loss = mse(vals, vb)

                loss = policy_loss + value_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

        self.net.eval()

    # -----------------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------------
    def save(self, path):
        torch.save({
            "model_state_dict": self.net.state_dict()
        }, path)

    @classmethod
    def load(cls, path, encoder: StateEncoder, n_simulations=400, c_puct=1.25,
             device=None, verbose=False, input_dim=250, hidden_dim=256, use_mcts=True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = Net(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        ckpt = torch.load(path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()

        # create and return a new agent instance
        return cls(
            net=net,
            encoder=encoder,
            n_simulations=n_simulations,
            c_puct=c_puct,
            device=device,
            use_mcts=use_mcts,
            verbose=verbose
        )

import time

# -------------------------
# Self-play training loop helper
# -------------------------
def self_play_training(agent: AlphaZeroMCTSAgent, game_factory,
                       n_games=5000,          # increased number of games
                       train_every=10,        # train more frequently
                       save_path="az_agent.pth", verbose=True):
    """
    game_factory: callable that returns a fresh Game() instance with agent set to None initially.
    The Game.play_game() method is not used; we use agent.play_game_and_collect which runs its own loop.
    """
    total_examples = 0
    start_time = time.time()   # <-- start timer

    for g in range(1, n_games + 1):
        game = game_factory()
        examples, (winner, turns) = agent.play_game_and_collect(game)
        agent.add_to_buffer(examples)
        total_examples += len(examples)

        if verbose and (g % 50 == 0 or g == 1):
            elapsed = time.time() - start_time
            avg_time_per_game = elapsed / g
            remaining_games = n_games - g
            est_remaining = remaining_games * avg_time_per_game

            # simple mm:ss formatting
            est_minutes = int(est_remaining // 60)
            est_seconds = int(est_remaining % 60)
            print(f"[SelfPlay] Game {g}/{n_games} winner={winner} turns={turns} examples={len(examples)}")
            print(f"          Estimated time left: {est_minutes:02d}m {est_seconds:02d}s")


        if g % train_every == 0:
            # train from buffer
            agent.train_from_buffer(batch_size=128, epochs=3, lr=1e-3)
            if verbose:
                print(f"[Train] Trained on buffer size={len(agent.train_buffer)} total_examples={total_examples}")
            # save snapshot
            agent.save(save_path)
            if verbose:
                print(f"[Save] Saved agent to {save_path}")

    # final save
    agent.save(save_path)
    if verbose:
        print(f"[Done] Training complete. Saved final model to {save_path}")

# -------------------------
# Example wiring / usage
# -------------------------
if __name__ == "__main__":
    try:
        from game import Game  # replace with real module / path if needed
        from expert_params import expert_params
    except Exception:
        Game = None

    if Game is None:
        print("Example: replace `your_game_module` import with the module that defines Game & Bot.")
    else:
        # richer encoder
        encoder = StateEncoder(max_history=40, expert_history_window=5)

        game0 = Game(game_target=25, expert_params=expert_params)
        dummy_vec = encoder.encode(game0.bot, game0)
        input_dim = dummy_vec.shape[0]

        net = Net(input_dim=input_dim, hidden_dim=256)
        agent = AlphaZeroMCTSAgent(
            net=net,
            encoder=encoder,
            n_simulations=400,  # more simulations
            c_puct=1.25,
            use_mcts=True
        )

        def game_factory():
            return Game(game_target=25, expert_params=expert_params)

        print(agent.device)
        # Larger training run with CUDA if available
        self_play_training(
            agent,
            game_factory,
            n_games=200,     # you can crank this higher
            train_every=3,   # training fairly often
            save_path="az_agent_2.pth"
        )

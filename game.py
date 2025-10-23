import numpy as np
import matplotlib.pyplot as plt
from bot import Bot

class Game:
    def __init__(self, game_target, expert_params, random_player=False, figure_ind=1):
        self.user_strokes = []
        self.user_strokes_same_diff = []
        self.user_win_loss = []

        self.bot_strokes = []
        self.bot_strokes_same_diff = []
        self.bot_win_loss = []

        self.user_grade = 0
        self.bot_grade = 0
        self.turn_number = 1
        self.game_target = game_target

        self.stop_game_flag = False
        self.cheating_flag = False
        self.random_player = random_player
        self.figure_ind = figure_ind
        self.bot = Bot(expert_params)
        self.grades_vs_turns = [[0], [0]]

    def play_game(self):
        print("Can you beat the machine?")
        print("Use 'a' for left, 'd' for right. Press 'q' to quit, 'c' to cheat.")
        self.draw_status()

        while self.user_grade < self.game_target and self.bot_grade < self.game_target:
            self.bot, bot_move = self.bot.bot_play(self)
            self.bot_strokes.append(bot_move)

            if self.cheating_flag:
                self.draw_bot_status()

            self.user_play()
            if self.stop_game_flag:
                break

            self.update_status()
            if not self.random_player:
                self.draw_status()

        winner = "You" if self.user_grade > self.bot_grade else "Bot"
        print(f"{winner} won after {self.turn_number} turns!")
        # Draw final plots and keep them open
        self.draw_status(final=True)
        self.draw_bot_status(final=True)

    def update_status(self):
        t = self.turn_number
        if t == 1:
            self.user_strokes_same_diff.append(1)
            self.bot_strokes_same_diff.append(1)
        else:
            us_diff = 1 if self.user_strokes[t-1] == self.user_strokes[t-2] else -1
            bs_diff = 1 if self.bot_strokes[t-1] == self.bot_strokes[t-2] else -1
            self.user_strokes_same_diff.append(us_diff)
            self.bot_strokes_same_diff.append(bs_diff)

        if self.bot_strokes[t-1] == self.user_strokes[t-1]:
            self.bot_grade += 1
            self.user_win_loss.append(-1)
            self.bot_win_loss.append(1)
        else:
            self.user_grade += 1
            self.user_win_loss.append(1)
            self.bot_win_loss.append(-1)

        self.grades_vs_turns[0].append(self.user_grade)
        self.grades_vs_turns[1].append(self.bot_grade)
        self.turn_number += 1

    def draw_status(self, final = False):
        # Create figure only if it doesn't exist
        if not hasattr(self, 'fig_status'):
            self.fig_status, self.ax_status = plt.subplots(figsize=(6, 4))
        self.ax_status.clear()

        # Bar plot of User and Bot scores
        self.ax_status.bar(["User", "Bot"], [self.user_grade, self.bot_grade], color=["b", "r"])
        self.ax_status.set_ylim(0, self.game_target)
        self.ax_status.set_title(f"Turn #{self.turn_number}")

        if final:
            plt.show()  # Keep the figure open
        else:
            plt.pause(0.1)  # Update during the game without blocking

    def draw_bot_status(self, final = False):
        bot = self.bot
        expert_types = bot.current_bot_status["experts_labels"]
        experts = bot.current_bot_status["experts"]

        # ---- User/Bot scores & error rate ----
        if not hasattr(self, 'fig_bot'):
            self.fig_bot, self.axs_bot = plt.subplots(3, 2, figsize=(12, 8))
            self.fig_bot_scores, self.axs_bot_scores = plt.subplots(2, 1, figsize=(10, 6))
            self.fig_bot_pred, self.ax_bot_pred = plt.subplots(figsize=(6, 4))

        # --- Scores ---
        self.axs_bot_scores[0].clear()
        turns = range(1, len(self.grades_vs_turns[0]) + 1)
        self.axs_bot_scores[0].plot(turns, self.grades_vs_turns[0], label="User")
        self.axs_bot_scores[0].plot(turns, self.grades_vs_turns[1], label="Bot")
        self.axs_bot_scores[0].legend()
        self.axs_bot_scores[0].set_ylabel("Score")

        # --- Error rate ---
        self.axs_bot_scores[1].clear()
        error_rate = [1 - (np.mean(self.bot_win_loss[:i+1]) + 1)/2 for i in range(len(self.bot_win_loss))]
        self.axs_bot_scores[1].plot(error_rate)
        self.axs_bot_scores[1].set_ylabel("Error rate")
        self.axs_bot_scores[1].set_ylim(0, 1)

        # --- Expert weights ---
        if experts.shape[2] > 0:
            last_turn_idx = experts.shape[2] - 1
            for i, label in enumerate(expert_types):
                self.axs_bot[i//2, i%2].clear()
                weights = experts[i, :, last_turn_idx]
                m_values = np.arange(len(weights))
                mask = weights != 0
                self.axs_bot[i//2, i%2].bar(m_values[mask], weights[mask])
                self.axs_bot[i//2, i%2].set_xlabel("Memory length m")
                self.axs_bot[i//2, i%2].set_ylabel("Weight")
                self.axs_bot[i//2, i%2].set_title(label)

        # --- Bot prediction ---
        self.ax_bot_pred.clear()
        qt = bot.current_bot_status["dec"]
        self.ax_bot_pred.bar([0], [qt], width=0.4)
        self.ax_bot_pred.set_ylim(-1, 1)
        self.ax_bot_pred.set_ylabel("Bot prediction probability")
        self.ax_bot_pred.set_title("Next prediction probability of the bot")
        self.ax_bot_pred.set_xticks([0])
        self.ax_bot_pred.set_xticklabels(["Prediction"])

        # --- Refresh all plots ---
        self.fig_bot_scores.tight_layout()
        self.fig_bot.tight_layout()
        self.fig_bot_pred.tight_layout()
        if final:
            plt.show()  # Keep the figure open
        else:
            plt.pause(0.1)  # Update during the game without blocking

    def user_play(self):
        if self.random_player:
            self.user_strokes.append(2 * np.random.binomial(1, 0.5) - 1)
            return

        while True:
            stroke = input("Your move [a/d/q/c]: ").strip().lower()
            if stroke == "d":
                self.user_strokes.append(1)
                break
            elif stroke == "a":
                self.user_strokes.append(-1)
                break
            elif stroke == "c":
                self.cheating_flag = not self.cheating_flag
                if self.cheating_flag:
                    self.draw_bot_status()
            elif stroke == "q":
                self.stop_game_flag = True
                break

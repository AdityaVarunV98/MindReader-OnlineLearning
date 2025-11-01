import numpy as np
from bot import Bot
import copy

class Game:
    def __init__(self, game_target, expert_params, agent=None, figure_ind=1, bot_memory_reset=True, benchmark_mode=False):
        """
        agent: an Agent object with act(game_state) function. If None, human input is used.
        bot_memory_reset: If True, new Bot() each game. If False, same bot continues learning across games.
        benchmark_mode: If True, no plotting or printing.
        """
        self.agent = agent
        self.bot_memory_reset = bot_memory_reset
        self.benchmark_mode = benchmark_mode

        self.expert_params = expert_params  # store to reset bot if needed
        self.bot = Bot(expert_params)

        self.figure_ind = figure_ind
        self.game_target = game_target

        # Game state
        self.reset_game_state()

    def reset_game_state(self):
        self.user_strokes = []
        self.user_strokes_same_diff = []
        self.user_win_loss = []

        self.bot_strokes = []
        self.bot_strokes_same_diff = []
        self.bot_win_loss = []

        self.user_grade = 0
        self.bot_grade = 0
        self.turn_number = 1

        self.stop_game_flag = False
        self.cheating_flag = False

        self.grades_vs_turns = [[0], [0]]

    def play_game(self):
        if not self.benchmark_mode:
            print("Game Started! Target =", self.game_target)

        # Reset bot if enabled
        if self.bot_memory_reset:
            self.bot = Bot(self.expert_params)

        while self.user_grade < self.game_target and self.bot_grade < self.game_target:
            # Bot move
            self.bot, bot_move = self.bot.bot_play(self)
            self.bot_strokes.append(bot_move)

            # Agent move (or human input)
            user_move = self.get_user_move()
            if user_move is None:  # user quit
                self.stop_game_flag = True
                break
            self.user_strokes.append(user_move)

            # Update scores
            self.update_status()

        winner = "user" if self.user_grade > self.bot_grade else "bot"
        turns_taken = self.turn_number - 1

        if not self.benchmark_mode:
            print(f"{winner.upper()} won in {turns_taken} turns!")

        return winner, turns_taken

    def get_user_move(self):
        if self.agent is not None:  # agent mode
            bot_copy = copy.deepcopy(self.bot)
            game_copy = copy.deepcopy(self)
            
            return self.agent.act({
                "bot": bot_copy,
                "game": game_copy
            })
        else:
            # human input mode
            while True:
                stroke = input("Your move [a/d/q]: ").strip().lower()
                if stroke == "d":
                    return 1
                elif stroke == "a":
                    return -1
                elif stroke == "q":
                    return None

    def update_status(self):
        t = self.turn_number
        # Same/diff updates
        if t == 1:
            self.user_strokes_same_diff.append(1)
            self.bot_strokes_same_diff.append(1)
        else:
            us_diff = 1 if self.user_strokes[t-1] == self.user_strokes[t-2] else -1
            bs_diff = 1 if self.bot_strokes[t-1] == self.bot_strokes[t-2] else -1
            self.user_strokes_same_diff.append(us_diff)
            self.bot_strokes_same_diff.append(bs_diff)

        # Grade update
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

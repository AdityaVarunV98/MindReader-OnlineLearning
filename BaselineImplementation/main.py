from game import Game
from expert_params import expert_params

if __name__ == "__main__":
    game_target = 20
    random_player = False  # set True for random
    figure_ind = 1

    g = Game(game_target, expert_params, random_player, figure_ind)
    g.play_game()

import numpy as np

class BiasDetector:
    def __init__(self, bias_memory, same_diff):
        self.bias_memory = bias_memory
        self.same_diff = same_diff
        self.predictions = []

    def predict(self, user_strokes, user_strokes_same_diff, turn_number):
        if turn_number == 1:
            bot_play = 0
        else:
            if self.same_diff == 0:
                target = user_strokes[-self.bias_memory:] if len(user_strokes) > self.bias_memory else user_strokes
                bot_play = np.mean(target)
            else:
                target = user_strokes_same_diff[-self.bias_memory:] if len(user_strokes_same_diff) > self.bias_memory else user_strokes_same_diff
                bot_play = np.mean(target) * user_strokes[-1]
        self.predictions.append(bot_play)
        return self, bot_play

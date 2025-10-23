import numpy as np

class PatternDetector:
    def __init__(self, pattern_length, same_diff):
        self.pattern_length = pattern_length
        self.same_diff = same_diff
        self.predictions = []

    def predict(self, user_strokes, user_strokes_same_diff, turn_number):
        if turn_number <= self.pattern_length:
            bot_play = 0
        else:
            target = user_strokes if self.same_diff == 0 else user_strokes_same_diff
            bot_play = self._pat_det(target)
            if self.same_diff == 1:
                bot_play *= user_strokes[-1]
        self.predictions.append(bot_play)
        return self, bot_play

    def _pat_det(self, target):
        pat = target[-self.pattern_length:]
        pat_grade = 0
        bot_play = pat[0]
        c = len(target) - self.pattern_length
        while c > 0:
            pat = np.roll(pat, 1)
            if np.sum(np.abs(target[c:c+self.pattern_length] - pat)) == 0:
                pat_grade += 1
            else:
                break
            c -= 1
        pat_grade = min(pat_grade, 2*self.pattern_length) / (2*self.pattern_length)
        return bot_play * pat_grade

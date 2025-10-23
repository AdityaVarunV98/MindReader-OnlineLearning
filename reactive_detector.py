import numpy as np

class ReactiveDetector:
    def __init__(self, memory_length):
        self.memory_length = memory_length
        self.predictions = []
        self.state_machine = np.zeros(2 ** (2 * memory_length + 1))

    def predict(self, user_strokes, user_win_loss, user_strokes_same_diff, turn_number):
        if turn_number <= self.memory_length + 2:
            bot_play = 0
        else:
            self, bot_play = self._reactive_det(user_strokes_same_diff, user_win_loss)
            bot_play *= user_strokes[-1]
        self.predictions.append(bot_play)
        return self, bot_play

    def _reactive_det(self, target, target_win_loss):
        ml = self.memory_length
        ind_map = 2 ** np.arange(2 * ml, -1, -1)

        # last_state = np.concatenate([target_win_loss[-ml-1:-1], target[-ml:]])
        # last_idx = int(np.sum(((last_state + 1)/2) * ind_map))
        # last_result = target[-1]
        
        if ml == 0:
            last_state = np.array([target_win_loss[-1]])
            last_idx = int((last_state[-1] + 1) / 2)
            last_result = target[-1]
        else:
            last_state = np.concatenate([target_win_loss[-ml-1:], target[-ml:]])
            last_idx = int(np.sum(((last_state + 1)/2) * ind_map))
            last_result = target[-1]

        if self.state_machine[last_idx] == 0:
            self.state_machine[last_idx] = 0.3 * last_result
        elif self.state_machine[last_idx] * last_result == 0.3:
            self.state_machine[last_idx] = 0.8 * last_result
        elif self.state_machine[last_idx] * last_result == 0.8:
            self.state_machine[last_idx] = 1.0 * last_result
        else:
            self.state_machine[last_idx] = 0

        # current_state = np.concatenate([target_win_loss[-ml:], target[-ml+1:]])
        # current_idx = int(np.sum(((current_state + 1)/2) * ind_map))

        if ml == 0:
            current_state = np.array([target_win_loss[-1]])
            current_idx = int((current_state[-1] + 1) / 2)
        else:
            current_state = np.concatenate([target_win_loss[-ml-1:], target[-ml:]])
            current_idx = int(np.sum(((current_state + 1)/2) * ind_map))

        return self, self.state_machine[current_idx]

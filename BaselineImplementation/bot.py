import numpy as np
from bias_detector import BiasDetector
from pattern_detector import PatternDetector
from reactive_detector import ReactiveDetector

class Bot:
    def __init__(self, expert_params):
        self.bias_detectors = [BiasDetector(m, 0) for m in expert_params["bias_memories"]]
        self.bias_detectors_same_diff = [BiasDetector(m, 1) for m in expert_params["bias_memories_same_diff"]]
        self.pattern_detectors = [PatternDetector(m, 0) for m in expert_params["pattern_length"]]
        self.pattern_detectors_same_diff = [PatternDetector(m, 1) for m in expert_params["pattern_length_same_diff"]]
        self.reactive_user_detectors = [ReactiveDetector(m) for m in expert_params["reactive_user_length"]]
        self.reactive_bot_detectors = [ReactiveDetector(m) for m in expert_params["reactive_bot_length"]]

        self.N = sum(len(v) for v in expert_params.values())
        self.current_bot_status = {"experts": np.zeros((6, max(len(v) for v in expert_params.values()), 0)),
                                   "dec": 0,
                                   "experts_labels": ["Bias", "Bias sd", "Pattern", "Pattern sd", "Reactive user", "Reactive bot"]}

    def bot_play(self, game):
        X = []

        for det in self.bias_detectors:
            _, p = det.predict(game.user_strokes, game.user_strokes_same_diff, game.turn_number)
            X.append(p)
        for det in self.bias_detectors_same_diff:
            _, p = det.predict(game.user_strokes, game.user_strokes_same_diff, game.turn_number)
            X.append(p)
        for det in self.pattern_detectors:
            _, p = det.predict(game.user_strokes, game.user_strokes_same_diff, game.turn_number)
            X.append(p)
        for det in self.pattern_detectors_same_diff:
            _, p = det.predict(game.user_strokes, game.user_strokes_same_diff, game.turn_number)
            X.append(p)
        for det in self.reactive_user_detectors:
            _, p = det.predict(game.user_strokes, game.user_win_loss, game.user_strokes_same_diff, game.turn_number)
            X.append(p)
        for det in self.reactive_bot_detectors:
            _, p = det.predict(game.bot_strokes, game.bot_win_loss, game.bot_strokes_same_diff, game.turn_number)
            X.append(p)

        X = np.array(X)
        if game.turn_number > 1:
            eta = np.sqrt(np.log(self.N) / (2 * game.game_target - 1))
            _, qt = self.aggregate_experts(X, np.array(game.user_strokes), eta)
        else:
            qt = 0

        bot_move = 2 * np.random.binomial(1, (qt + 1) / 2) - 1
        return self, bot_move

    # def aggregate_experts(self, X, user_strokes, eta):
    #     yt_all = []
    #     yt_mat = []

    #     def loss(pred):
    #         return np.exp(-eta * np.sum(np.abs(np.array(pred[:-1]) - user_strokes)))

    #     for group in [self.bias_detectors, self.bias_detectors_same_diff,
    #                   self.pattern_detectors, self.pattern_detectors_same_diff,
    #                   self.reactive_user_detectors, self.reactive_bot_detectors]:
    #         group_weights = [loss(det.predictions) for det in group]
    #         yt_all.extend(group_weights)
    #         yt_mat.append(group_weights)

    #     yt_all = np.array(yt_all)
    #     yt_all /= np.sum(yt_all)
    #     qt = np.dot(yt_all, X)

    #     yt_mat_np = np.zeros((6, max(len(g) for g in [self.bias_detectors, self.bias_detectors_same_diff,
    #                                                  self.pattern_detectors, self.pattern_detectors_same_diff,
    #                                                  self.reactive_user_detectors, self.reactive_bot_detectors])))
    #     for i, arr in enumerate(yt_mat):
    #         yt_mat_np[i, :len(arr)] = arr / np.sum(yt_all)
    #     self.current_bot_status["experts"] = np.concatenate(
    #         (self.current_bot_status["experts"], yt_mat_np[:, :, None]), axis=2)
    #     self.current_bot_status["dec"] = qt
    #     return self, qt

    def aggregate_experts(self, X, user_strokes, eta):
        yt_all = []
        yt_mat = []

        def loss(pred):
            if len(pred) < 2:  # no history yet
                return 0
            return np.exp(-eta * np.sum(np.abs(np.array(pred[:-1]) - user_strokes)))

        for group in [self.bias_detectors, self.bias_detectors_same_diff,
                    self.pattern_detectors, self.pattern_detectors_same_diff,
                    self.reactive_user_detectors, self.reactive_bot_detectors]:
            group_weights = [loss(det.predictions) for det in group if len(det.predictions) > 0]
            yt_all.extend(group_weights)
            yt_mat.append(group_weights)

        if len(yt_all) == 0:
            qt = 0
        else:
            yt_all = np.array(yt_all)
            yt_all /= np.sum(yt_all)
            qt = np.dot(yt_all, X[:len(yt_all)])  # only include experts that exist

        yt_mat_np = np.zeros((6, max(len(g) for g in [self.bias_detectors, self.bias_detectors_same_diff,
                                                    self.pattern_detectors, self.pattern_detectors_same_diff,
                                                    self.reactive_user_detectors, self.reactive_bot_detectors])))
        for i, arr in enumerate(yt_mat):
            if len(arr) > 0:
                yt_mat_np[i, :len(arr)] = np.array(arr) / np.sum(yt_all)

        # Append latest weights for dashboard
        self.current_bot_status["experts"] = np.concatenate(
            (self.current_bot_status["experts"], yt_mat_np[:, :, None]), axis=2)
        self.current_bot_status["dec"] = qt
        return self, qt


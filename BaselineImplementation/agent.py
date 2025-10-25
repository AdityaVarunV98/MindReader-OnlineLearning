import numpy as np

class BaseAgent:
    """
    Base class for agents. Any agent must implement:
        act(game_state) -> returns -1 (left) or +1 (right)
    """
    def __init__(self, name="BaseAgent"):
        self.name = name

    def act(self, game_state):
        raise NotImplementedError("Agent must implement act().")


class RandomAgent(BaseAgent):
    """Plays randomly with equal probability"""
    def __init__(self):
        super().__init__(name="RandomAgent")

    def act(self, game_state):
        return 2 * np.random.binomial(1, 0.5) - 1


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

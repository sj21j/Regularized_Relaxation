from typing import List

class Result:
    def __init__(self, best_loss: float, best_string: str, losses: List[float], strings: List[str]):
        self.best_loss = best_loss
        self.best_string = best_string
        self.losses = losses
        self.strings = strings

    def to_dict(self):
        return {
            "best_loss": self.best_loss,
            "best_string": self.best_string,
            "losses": self.losses,
            "strings": self.strings
        }

class RRResult:
    def __init__(self, best_losses: List[float], best_strings: List[str], losses: List[float], strings: List[str], distances: List[float]):
        self.best_losses = best_losses
        self.best_strings = best_strings
        self.losses = losses
        self.strings = strings
        self.distances = distances

    def to_dict(self):
        return {
            "best_losses": self.best_losses,
            "best_strings": self.best_strings,
            "losses": self.losses,
            "strings": self.strings,
            "distances": self.distances
        }
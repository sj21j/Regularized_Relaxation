class Result:
    def __init__(self, best_loss: float, best_string: str, losses: list[float], strings: list[str]):
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
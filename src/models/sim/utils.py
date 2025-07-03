
class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def __call__(self, val = None):
        if val is not None:
            self.sum += val
            self.count += 1
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0
        
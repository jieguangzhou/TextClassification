import time
import datetime

class Timer:
    def __init__(self):
        self.poit = {}
        self.last_key = '__init__'
        self.poit[self.last_key] = self.now

    def cost_time(self, key=None):
        key = key if key is not None else '__init__'
        return self.now - self.poit[key]

    def mark(self, key, cost_time_key=None):
        now = self.now
        cost_time = self.cost_time(cost_time_key)
        self.poit[key] = now
        self.last_key = key
        return cost_time


    @property
    def now(self):
        return datetime.datetime.now()

if __name__ == '__main__':
    timer = Timer()
    time.sleep(1)
    print(timer.now)
    print(timer.cost_time())
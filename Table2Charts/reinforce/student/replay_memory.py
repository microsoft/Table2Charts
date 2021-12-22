import random
from threading import Lock


class ReplayMemory:
    # TODO: implement prioritised replay buffer https://github.com/rlcode/per

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.lock = Lock()

    def push(self, experience):
        with self.lock:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = experience

            self.position += 1
            if self.position == self.capacity:
                self.position = 0

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

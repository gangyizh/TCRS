
import random
from collections import namedtuple, deque
from typing import List

Experience = namedtuple("Experience", ["user_id", "attribute_sequence_ids", 'next_attribute_sequence_ids', 'a_logprob', "action_embs", "reward", "done", "rec_done"])




class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args) -> None:
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def get_all(self) -> List[Experience]:
        return list(self.buffer)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer.clear()
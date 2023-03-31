import random
from collections import namedtuple, deque

# Define a named tuple to store experiences
Transition = namedtuple(
    "Transition",
    ('state', 'action', 'next_state', 'reward'),
)


class ReplayBuffer:
    """
    ReplayBuffer class stores experiences, each of which consists of:
    - the current observation,
    - the action taken,
    - the reward received,
    - the next state,
    - a flag indicating whether the episode is finished.
    - the hidden state
    - the cell state

    The class provides methods to:
    - add(): add new experiences to the buffer
    - sample(): sample a batch of experiences for training
    - len(): retrieve the number of experiences currently stored in the buffer.
    - clear(): empty the buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

    def add(self, *args):
        """
        Description:
        ------------
        Adds a new experience to the buffer.

        Args:
        -----
        obs (np.array): current observation
        action (np.array): action taken
        reward (float): reward received
        next_obs (np.array): next observation
        done (bool): flag indicating whether the episode is finished

        Returns:
        --------
        None
        """
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """
        Description:
        ------------
        Samples a batch of experiences from the buffer.

        Args:
        -----
        batch_size (int): batch size

        Returns:
        --------
        batch (list): list of experiences
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
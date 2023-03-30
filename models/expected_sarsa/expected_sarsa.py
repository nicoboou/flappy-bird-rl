import numpy as np

from models.base_agent import BaseAgent


class ExpectedSarsaAgent(BaseAgent):
    """
    Expected SARSA agent.

    Attributes:
    ----------
        num_actions (int): The number of actions.
        num_states (int): The number of states.
        epsilon (float): The epsilon parameter for exploration.
        step_size (float): The step-size.
        discount (float): The discount factor.
        rand_generator (RandomState): The random number generator.
        q (numpy array): The action-value function.
        prev_state (int): The previous state.
        prev_action (int): The previous action.
    """

    def __init__(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }

        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        # Create an array for action-value estimates and initialize it to zero.
        if agent_init_info["policy"]:
            self.q = agent_init_info["policy"]
        else:
            self.q = {}

        # The previous state and action
        self.prev_state = None
        self.prev_action = None

    def was_visited(self, state):
        if state not in self.q.keys():
            self.q[state] = {0: 0, 1: 0}

    def policy(self, observation):
        """
        Determine the policy corresponding to the final action-value function estimate.
        """

        if self.q[observation]:
            # Find key_value pair with highest Q value in the dictionary
            pi = max(self.q[observation], key=self.q[observation].get)
            print("Policy: ", pi)
        else:
            raise ValueError(
                "Policy empty for this state, state was probably not explored. \n Train your agent first"
            )

        return pi

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        self.was_visited(state)

        # Get the current q values.
        state_values = self.q[state]
        current_q = [state_values[0], state_values[1]]

        # Choose action using epsilon greedy.
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        self.prev_state = state
        self.prev_action = action

        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        self.was_visited(state)

        # Get the current q values.
        state_values = self.q[state]
        current_q = [state_values[0], state_values[1]]

        # Choose action using epsilon greedy.
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        # 1. Instantiate the expected_q variable
        expected_q = 0

        # 2. Get the q_max value
        q_max = np.max(current_q)

        # 3. Get the policy values
        pi = np.ones(self.num_actions) * self.epsilon / self.num_actions + (
            current_q == q_max
        ) * (1 - self.epsilon) / np.sum(current_q == q_max)

        # 4. Calculate the expected_q value
        expected_q = np.sum(current_q * pi)

        # 5. Update the q value
        self.q[self.prev_state][self.prev_action] += self.step_size * (
            reward
            + self.discount * expected_q
            - self.q[self.prev_state][self.prev_action]
        )

        # Update the previous state and action.
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # Perform the last update in the episode (1 line)
        self.q[self.prev_state][self.prev_action] += self.step_size * (
            reward - self.q[self.prev_state][self.prev_action]
        )

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for index, _ in enumerate(q_values):
            if q_values[index] > top:
                top = q_values[index]
                ties = []

            if q_values[index] == top:
                ties.append(index)

        return self.rand_generator.choice(ties)

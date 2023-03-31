import os
import pickle
import numpy as np
from tqdm import tqdm 
import wandb

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

    def __init__(self, env, config):
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
        # Store the parameters provided in config.
        self.env = env
        self.num_actions = config["NUM_ACTIONS"]
        self.num_steps = config["NUM_STEPS"]
        self.num_episodes = config["NUM_EPISODES"]
        self.discount = config["DISCOUNT"]
        self.epsilon = config["EPSILON"]
        self.step_size = config["STEP_SIZE"]
        self.seed = config["SEED"]
        self.rand_generator = np.random.RandomState(config["SEED"])

        # Create an array for action-value estimates and initialize it to zero.
        if config["POLICY"]:
            self.q = config["POLICY"]
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
        """
        Run when the agent terminates.
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

    def train(self, all_reward_sums, all_state_visits, all_scores):
        """
        Train the agent using a given environment.

        Args:
            all_reward_sums (dict): Dictionary containing the reward sums for each agent.
            all_state_visits (dict): Dictionary containing the state visits for each agent.
            all_scores (dict): Dictionary containing the scores for each agent.

        Returns:
            None
        """
        # Init wandb
        wandb.init(project="Flappy-RL")
        
        # Agent name
        agent_name = self.__class__.__name__

        # Iteration over the number of runs
        for run in tqdm(range(self.num_episodes)):

            # Set the seed value to the current run index
            self.seed = run

            # Initialize the environment
            # Returns (obs: (x_dist,y_dist), info: {"score", "player", "distance"})
            state, info = self.env.reset()

            # Set done to False
            done = False

            reward_sums = []
            state_visits = {}

            # Iterate over the number of episodes
            for step in range(self.num_steps):
                if step == 0:
            
                    # Keep track of the visited states
                    state, info = self.env.reset()
                    action = self.agent_start(state)

                    state_visits[state] = 1
                    state, reward, done, _, info = self.env.step(action)
                    reward_sums.append(reward)

                else:
                    while not done:
                        action = self.agent_step(reward, state)

                        if state not in state_visits: 
                            state_visits[state] = 1
                        else:
                            state_visits[state] += 1

                        state, reward, done, _, info = self.env.step(action)
                        reward_sums.append(reward)

                        # If terminal state
                        if done:
                            self.agent_end(reward)
                            break

            all_reward_sums[agent_name].append(np.sum(reward_sums))
            all_state_visits[agent_name].append(np.sum(state_visits))
            all_scores[agent_name].append(np.sum(info["score"]))

            wandb.log({"episode_reward": np.sum(reward_sums), "episode_score":np.sum(info["score"]), "episode_duration": step+1})

        # Print the average reward sum & average score after training
        print("Average Reward Sum: ", np.mean(all_reward_sums[agent_name]))
        print("Average Score: ", np.mean(all_scores[agent_name]))

        wandb.log({"average_reward_sum": np.mean(all_reward_sums[agent_name]), "average_score": np.mean(all_scores[agent_name])})
        
        wandb.finish()

        # Create dir if not exists
        AGENT_DIR = 'models/' + agent_name + '/'
        if not os.path.exists(AGENT_DIR):
            os.makedirs(AGENT_DIR)

        # Save policy for simulation
        with open(AGENT_DIR + agent_name + '_q_values.pickle', 'wb') as handle:
            pickle.dump(self.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save results for vizualization
        with open(AGENT_DIR + agent_name + '_results.pickle', 'wb') as handle:
            pickle.dump(all_reward_sums[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_state_visits[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_scores[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
            


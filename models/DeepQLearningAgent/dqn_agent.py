import os
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm 
import random
import itertools
import wandb

from models.base_agent import BaseAgent
from models.DeepQLearningAgent.experience_replay import ReplayBuffer
from models.DeepQLearningAgent.q_network import DeepQNetwork

class DeepQLearningAgent(BaseAgent):

    def __init__(self, env, config):
        # Store the parameters provided in config.
        self.env = env
        self.num_actions = config["NUM_ACTIONS"]
        self.num_episodes = config["NUM_EPISODES"]
        self.discount = config["DISCOUNT"]
        self.seed = config["SEED"]
        self.rand_generator = np.random.RandomState(config["SEED"])

        # State size
        self.state_size = config["STATE_SIZE"]

        # Action size
        self.action_size = config["NUM_ACTIONS"]

        # Learning rate
        self.learning_rate = config["LR"]

        # Update rate of the target network
        self.tau = config["TAU"]

        # Number of transitions sampled from the replay buffer for each training step
        self.batch_size = config["BATCH_SIZE"]

        # Epsilon
        self.epsilon_decay = config["EPSILON_DECAY"]
        self.epsilon_start = config["EPSILON_START"]
        self.epsilon_end = config["EPSILON_END"]

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")

        # Instanciate Replay Buffer for experience replay
        self.replay_buffer = ReplayBuffer(config["REPLAY_BUFFER_SIZE"])

        # Instanciate Q-Network
        self.q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)

        # Instanciate Target Q-Network
        self.target_q_network = DeepQNetwork(self.state_size, self.action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network .state_dict())

        # Define loss function
        self.criterion = nn.SmoothL1Loss()

        # Define optimizer
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.learning_rate, amsgrad=True)

        # Define train episodes durations for plot
        self.train_episodes_durations = []

        # Initiate counter
        self.steps_done = 0

        # The previous state and action
        self.prev_state = None
        self.prev_action = None

    def policy(self, observation):
        with torch.no_grad():
                return self.target_q_network.forward(observation).max(1)[1].view(1, 1) # Pick index of max value to return action , shape: (1,)

    def epsilon_greedy_select_action(self, state):
        """ 
        Select an action based on the current state and the epsilon-greedy policy.

        Parameters
        ----------
        state (torch.Tensor): the current state of the environment.

        Returns
        -------
        action (torch.Tensor): the action to be taken.
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.target_q_network.forward(state).max(1)[1].view(1, 1) # Pick index of max value to return action , shape: (1,)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    def train(self, all_reward_sums, all_state_visits, all_scores):
        """
        Description:
        ------------
        Train the agent using the Deep Q-Learning algorithm.

        Parameters:
        -----------
        all_reward_sums (list): list of all reward sums.
        all_state_visits (list): list of all state visits.
        all_scores (list): list of all scores.

        Returns:
        --------
        None
        """
        # Init wandb
        wandb.init(project="Flappy-RL")
        
        # Agent name
        agent_name = self.__class__.__name__

        for episode in tqdm(range(self.num_episodes)):

            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32,device=self.device).unsqueeze(0)

            # Instanciate variables
            losses = []
            episode_reward = 0
            done = False

            state_visits = {}

            for step in itertools.count():

                # Select actionf
                action = self.epsilon_greedy_select_action(state)

                # Perform step
                observation, reward, done, _, info = self.env.step(action.item())

                # Counter for states visits
                if observation not in state_visits: 
                    state_visits[state] = 1
                else:
                    state_visits[state] += 1

                # Update the episode reward
                episode_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(np.array(observation), dtype=torch.float32, device=self.device).unsqueeze(0)

                # Convert reward to tensor
                reward = torch.tensor([reward], device=self.device)
                
                # Store the transition in memory
                self.replay_buffer.add(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.train_batch(batch)
                    losses.append(loss)

                # Soft update of the target network's weights
                self.update_target_q_network()

                if done:
                    self.train_episodes_durations.append(step + 1)
                    break
                
            all_reward_sums[agent_name].append(episode_reward)
            all_state_visits[agent_name].append(np.sum(state_visits))
            all_scores[agent_name].append(np.sum(info["score"]))

            loss_mean = np.array(losses).mean()
            wandb.log({"episode_reward": episode_reward, "episode_score":np.sum(info["score"]), "loss": loss_mean, "episode_duration": step+1})
            # print(
            #     f"Episode {episode:05d} | Episode steps: {step+1} | Episode reward: {episode_reward} | Loss: {loss_mean:.4f}"
            # )

            # Decay epsilon
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        # Print the average reward sum & average score after training
        print("-"*50)
        print("Average Reward Sum: ", np.mean(all_reward_sums[agent_name]))
        print("Average Score: ", np.mean(all_scores[agent_name]))

        wandb.log({"average_reward_sum": np.mean(all_reward_sums[agent_name]), "average_score": np.mean(all_scores[agent_name])})
        wandb.finish()

        # Create dir if not exists
        AGENT_DIR = 'models/' + agent_name + '/'
        if not os.path.exists(AGENT_DIR):
            os.makedirs(AGENT_DIR)

        # Save network weights
        torch.save(self.q_network.state_dict(), AGENT_DIR + agent_name + '_model.pt')

        # Save results for vizualization
        with open(AGENT_DIR + agent_name + '_results.pickle', 'wb') as handle:
            pickle.dump(all_reward_sums[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_state_visits[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_scores[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

        return np.mean(all_scores[agent_name])

    def update_target_q_network(self):
        """
        Description:
        ------------
        Update the target network by copying the parameters from the Q-Network.
        The target Q-Network is used to provide a stable target for the Q-learning algorithm during the training process.
        The Q-learning algorithm uses the Bellman equation to update the Q-values, which involves the Q-values from the next state.
        However, if we update the Q-network at each step using the same network to calculate the next state's Q-value, it can lead to instability or divergence.
        To mitigate this problem, we use a separate target Q-network with frozen parameters to calculate the Q-values for the next state during the update.
        We update the target network's parameters after a fixed number of steps to keep it synchronized with the Q-network.
        This technique is known as "fixed Q-targets" and is commonly used in deep reinforcement learning.

        Args:
        -----
        None

        Returns:
        --------
        None
        """

        # for target_param, param in zip(
        #     self.target_q_network.parameters(),
        #     self.q_network.parameters(),
        # ):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        
        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            
        self.target_q_network.load_state_dict(target_net_state_dict)


    def train_batch(self, batch):
        """
        Description:
        ------------
        Train a batch sampled with the replay buffer.

        Parameters:
        -----------
        batch (tuple): a batch of transitions sampled from the replay buffer.

        Returns:
        --------
        loss (torch.Tensor): the loss of the batch.
        """

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.q_network.forward(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_q_network.forward(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Zero the gradients
        self.optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)

        # Step over the optimizer
        self.optimizer.step()

        return loss.item()

    def load(self, path):
        """
        Load the model weights from a file.

        Args:
            path (str): Path to the file from where to load the model.
        """
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(torch.load(path))
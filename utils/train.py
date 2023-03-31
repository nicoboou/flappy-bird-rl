import os
import pickle
import numpy as np
from tqdm import tqdm 

def train(agent, env, all_reward_sums, all_state_visits, all_scores):
    """
    Train the agent using a given environment.

    Args:
        agent (Agent): The agent to train.
        env (Environment): The environment to train the agent in.
        all_reward_sums (dict): Dictionary containing the reward sums for each agent.
        all_state_visits (dict): Dictionary containing the state visits for each agent.
        all_scores (dict): Dictionary containing the scores for each agent.

    Returns:
        None
    """
    # Agent name
    agent_name = agent.__class__.__name__

    # Iteration over the number of runs
    for run in tqdm(range(agent.num_episodes)):

        # Set the seed value to the current run index
        agent.seed = run

        # Initialize the environment
        # Returns (obs: (x_dist,y_dist), info: {"score", "player", "distance"})
        state, info = env.reset()

        # Set done to False
        done = False

        reward_sums = []
        state_visits = {}

        # Iterate over the number of episodes
        for step in range(agent.num_steps):
            if step == 0:
        
                # Keep track of the visited states
                state, info = env.reset()
                action = agent.agent_start(state)

                state_visits[state] = 1
                state, reward, done, _, info = env.step(action)
                reward_sums.append(reward)

            else:
                while not done:
                    action = agent.agent_step(reward, state)

                    if state not in state_visits: 
                        state_visits[state] = 1
                    else:
                        state_visits[state] += 1

                    state, reward, done, _, info = env.step(action)
                    reward_sums.append(reward)

                    # If terminal state
                    if done:
                        action = agent.agent_end(reward)
                        break

        all_reward_sums[agent_name].append(np.sum(reward_sums))
        all_state_visits[agent_name].append(state_visits)
        all_scores[agent_name].append(info["score"])

        # Print the average reward sum & average score for the last 10 episodes
        if run % 1000 == 0:
            print("Run: ", run, "Average Reward Sum: ", np.mean(all_reward_sums[agent_name][-10:]))
            print("Run: ", run, "Average Score: ", np.mean(all_scores[agent_name][-10:]))

    # Create dir if not exists
    AGENT_DIR = 'models/' + agent_name + '/'
    if not os.path.exists(AGENT_DIR):
        os.makedirs(AGENT_DIR)

    # Save policy for simulation
    with open(AGENT_DIR + agent_name + '_q_values.pickle', 'wb') as handle:
        pickle.dump(agent.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save results for vizualization
    with open(AGENT_DIR + agent_name + '_results.pickle', 'wb') as handle:
        pickle.dump(all_reward_sums[agent_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
        

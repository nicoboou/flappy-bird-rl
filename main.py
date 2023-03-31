import os, sys
import argparse
import numpy as np
import gymnasium as gym
import time
import pickle
import json
import torch

import text_flappy_bird_gym
from models.ExpectedSarsaAgent.expected_sarsa_agent import ExpectedSarsaAgent
from models.DeepQLearningAgent.dqn_agent import DeepQLearningAgent


def main(args):

    # Initialize dictionaries for training results
    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {} # Contains state visit counts during the last 10 episodes
    all_scores = {} # Contains the scores obtained for each run

    parser = argparse.ArgumentParser()

    # Argument parser
    parser.add_argument(
        "-agent",
        dest="agent_type",
        type=str,
        default="expected_sarsa",
        required=False,
    )

    parser.add_argument(
        "-train",
        dest="train",
        type=bool,
        default=False,
        required=False,
    )

    args = parser.parse_args()

    # Instanciate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs, info = env.reset()
    env_info = {}

    # Import config file
    with open("./config/one_run_config.json", "r") as f:
        config = json.load(f)


    # Instanciate agent
    if args.agent_type == "expected_sarsa":
        agent_info = config["ExpectedSarsaAgent"]

        # If train mode is on, train the agent
        if args.train:
            agent = ExpectedSarsaAgent(config=agent_info, env=env)
            all_reward_sums[agent.__class__.__name__] = []
            all_state_visits[agent.__class__.__name__] = []
            all_scores[agent.__class__.__name__] = []
            print("Training started...")
            agent.train(all_reward_sums, all_state_visits, all_scores)
            print("Training finished. Saving model...")

        # Load the trained agent
        with open(
            "./models/ExpectedSarsaAgent/ExpectedSarsaAgent_q_values.pickle", "rb"
        ) as handle:
            agent_info["POLICY"] = pickle.load(handle)

        agent = ExpectedSarsaAgent(config=agent_info, env=env)
    
    elif args.agent_type == "dqn":
        agent_info = config["DeepQLearningAgent"]

        # If train mode is on, train the agent
        if args.train:
            agent = DeepQLearningAgent(config=agent_info, env=env)
            all_reward_sums[agent.__class__.__name__] = []
            all_state_visits[agent.__class__.__name__] = []
            all_scores[agent.__class__.__name__] = []
            print("Training started...")
            agent.train(all_reward_sums, all_state_visits, all_scores)
            print("Training finished. Saving model...")

        # Load the trained agent
        agent = DeepQLearningAgent(config=agent_info, env=env)
        agent.load(path="./models/DeepQLearningAgent/DeepQLearningAgent_model.pt")

    # iterate
    while True:

        print(f"Observation: {obs}")

        # Select next action
        if args.agent_type == "expected_sarsa":
            action = agent.policy(obs)
            # Appy action and return new observation of the environment
            obs, reward, done, _, info = env.step(action)

        elif args.agent_type == "dqn":
            obs = np.array(obs)
            obs = torch.tensor(obs, dtype=torch.float32,device=agent.device).unsqueeze(0)
            action = agent.policy(obs)
            # Appy action and return new observation of the environment
            obs, reward, done, _, info = env.step(action.item())

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2)  # FPS

        # If player is dead break
        if done:
            break

    env.close()


if __name__ == "__main__":
    main(sys.argv[1:])

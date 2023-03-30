import os, sys
import argparse
import numpy as np
import gymnasium as gym
import time
import pickle

import text_flappy_bird_gym
from models.expected_sarsa.expected_sarsa import ExpectedSarsaAgent


def main(args):

    parser = argparse.ArgumentParser()

    # Argument parser
    parser.add_argument(
        "-agent",
        dest="agent_type",
        type=str,
        default="expected_sarsa",
        required=False,
    )

    args = parser.parse_args()

    # Instanciate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs, info = env.reset()

    agent_info = {
        "num_actions": 2,
        "epsilon": 0.1,
        "epsilon_decay": 0.999,
        "step_size": 0.7,
        "step_size_decay": 0.9999,
        "discount": 1.0,
    }
    agent_info["seed"] = 0
    env_info = {}

    # Instanciate agent
    if args.agent_type == "expected_sarsa":
        with open(
            "./models/expected_sarsa/expected_sarsa_q_values.pickle", "rb"
        ) as handle:
            agent_info["policy"] = pickle.load(handle)

        agent = ExpectedSarsaAgent(agent_info)

    # iterate
    while True:

        print(f"Observation: {obs}")

        # Select next action
        action = agent.policy(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

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

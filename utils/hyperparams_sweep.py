# Import the W&B Python Library and log into W&B
import sys
import argparse
import gymnasium as gym
import wandb
import json

import text_flappy_bird_gym
from models.DeepQLearningAgent.dqn_agent import DeepQLearningAgent
from models.ExpectedSarsaAgent.expected_sarsa_agent import ExpectedSarsaAgent

all_reward_sums = {"DeepQLearningAgent": [], "ExpectedSarsaAgent": []} # Contains sum of rewards during episode
all_state_visits = {} # Contains state visit counts during the last 10 episodes
all_scores = {} # Contains the scores obtained for each run

def objective(agent, env, config):
    if agent == 'DQN':
        agent = DeepQLearningAgent(config=config,env=env)
        score = agent.train(all_reward_sums,all_state_visits,all_scores)
        return score
    
    elif agent == "ExpectedSARSA":
        agent = ExpectedSarsaAgent(config=config,env=env)
        score = agent.train(all_reward_sums,all_state_visits,all_scores)
        return score

def sweep(agent:str,env):
    wandb.init(project='Flappy-RL')
    score = objective(agent,env, wandb.config)
    wandb.log({'score': score})


def main(args):
    # Login to WandB
    wandb.login()

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

    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    env_info = {}
    
    # 2Define the search space
    if args.agent_type == "expected_sarsa":
        with open('../config/sweep_config_expected_sarsa.json', "r") as f:
            sweep_configuration = json.load(f)
    
    elif args.agent_type == "dqn":
        with open('../config/sweep_config_dqn.json', "r") as f:
            sweep_configuration = json.load(f) 

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Flappy-RL')
    wandb.agent(sweep_id, function=sweep, count=10)

if __name__ == "__main__":
    main(sys.argv[1:])

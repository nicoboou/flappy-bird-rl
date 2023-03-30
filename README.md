# Flappy Bird Gym with Reinforcement Learning

This project implements two different reinforcement learning algorithms for the Flappy Bird Gym environment. The two algorithms used are Expected SARSA and Deep Q Learning.

## Getting Started

To get started, you'll need to have the following installed:

- Python 3.x
- OpenAI Gym
- PyTorch

You can install the required packages using conda:
`conda create -f environment.yml`

## Running the Code

### Playing the Game

To run the code, simply clone the repository and navigate to the main directory. Then, run the following command:
`python3 main.py -agent deep_q_learning -env FlappyBird-v0 -render`

The above command will run the Deep Q Learning algorithm on the Flappy Bird Gym environment. The `-render` flag will render the game in your terminal. To run the Expected SARSA algorithm, simply replace `deep_q_learning` with `expected_sarsa`.

### Training the Agent

To train the agent, simply add the `-train` flag to the command above. For example, to train the Deep Q Learning agent, run the following command:
`python3 main.py -agent deep_q_learning -env FlappyBird-v0 -render -train`

The above command will train the Deep Q Learning agent for 1000 episodes. To train the Expected SARSA agent, simply replace `deep_q_learning` with `expected_sarsa`.

### Changing the Hyperparameters

To change the hyperparameters, simply edit the `config.py` file. The hyperparameters are as follows:

- `NUM_EPISODES`: The number of episodes to train the agent for.
- `NUM_RUNS`: The number of runs to average the results over.
- `STEP_SIZE`: The step size for the agent.
- `EPSILON`: The epsilon value for the epsilon-greedy policy.
- `EPSILON_DECAY`: The decay rate for the epsilon value.
- `EPSILON_MIN`: The minimum value for the epsilon value.
- `DISCOUNT`: The discount factor for the agent.
- `BATCH_SIZE`: The batch size for the Deep Q agent.
- `REPLAY_MEMORY_SIZE`: The size of the replay memory for the Deep Q agent.

## Results

After training is complete, you can also evaluate the performance of the algorithms by running the following command:

- Expected SARSA: `python3 evaluate.py -agent expected_sarsa -env FlappyBird-v0 -policy_path ./models/expected_sarsa/expected_sarsa.pickle`
- Deep QLearning: `python3 evaluate.py -agent deep_q_learning -env FlappyBird-v0 -policy_path ./models/deep_q_learning.pt`

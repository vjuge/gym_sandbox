import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0')
# env = gym.make('MountainCar-v0', render_mode="human")

# choose number of grid points to use
n_grid = 10
# grid over observations[0] and actions[1]
gridspace = {}
# empty qtable
q_table = np.zeros((n_grid, n_grid, env.action_space.n))
# learning rate
alpha = 0.1
# discount factor
gamma = 0.95

# number of episodes of training
epochs = 5000

# exploration/exploitation rate : 1 = 100% exploration, 0 = 100% exploitation
# 'epsilon greedy' strategy : epsilon is very high at the beginning, then decreases over time
epsilon = 1.

epsilon_min = 0.1
start_epsilon_decaying = 1000
end_epsilon_decaying = epochs // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

TRAIN = False
PREDICT = True


def init():
    """
    Initialize gridspace and q_table
    :return:
    """
    print("**** Initialization ****")
    global gridspace, q_table
    for ii in range(env.observation_space.shape[0]):
        gridspace[ii] = np.linspace(env.observation_space.low[ii], env.observation_space.high[ii], n_grid)

    q_table = np.random.uniform(low=-2, high=2, size=(n_grid, n_grid, env.action_space.n))


# define a function that makes it easy to get our discrete indices from an observation
def to_discrete_state(observation: np.ndarray(2, )):
    """
    returns a tuple of (position, velocity) discrete state
    :argument observation: ndarray of continuous observation state (position, velocity)
    """
    # print("** get_discretized_observation **")
    # print("to_discrete_state | observation: ", observation)
    # print("gridspace: ", gridspace)
    # for (ii, grid) in gridspace.items():
    #     print("ii: ", ii)
    #     print("grid: ", grid)
    #     print("observation[ii]: ", observation[ii])
    #     print("np.abs(grid - observation[ii]): ", np.abs(grid - observation[ii]))
    #     print("np.argmin(np.abs(grid - observation[ii])): ", np.argmin(np.abs(grid - observation[ii])))
    # return tuple of indices in the grid
    discrete_state = np.array([np.argmin(np.abs(grid - observation[ii])) for (ii, grid) in gridspace.items()])
    # print("discrete_state: ", discrete_state)
    # state = gridspace[0][index[0]], gridspace[1][index[1]]
    # print("to_discrete_state | state: ", tuple(discrete_state))
    return tuple(discrete_state)


def explore_or_exploit(q_value):
    """
    Returns an action according to an epsilon-greedy approach
    :param q_value:
    :return: 
    """
    if np.random.random() > epsilon:
        # exploit
        strategy = "exploit"
        action = np.argmax(q_value)
    else:
        # explore
        strategy = "explore"
        action = np.random.randint(0, env.action_space.n)

    # print("action: ", action, "strategy: ", strategy)
    return action


def train():
    """
    Train the agent
    :return:
    """
    global epsilon, q_table
    print("**** Training ****")
    for episode in range(epochs):
        discrete_state = to_discrete_state(env.reset()[0])
        done = False
        while not done:

            # choose action
            if np.random.random() > epsilon:
                # exploit
                action = np.argmax(q_table[discrete_state])
            else:
                # explore
                action = np.random.randint(0, env.action_space.n)

            observation, reward, terminated, truncated, info = env.step(action)
            new_discrete_state = to_discrete_state(observation)

            # if observation[0] >= env.goal_position:
            #     reward = 1
            # print("obs: ", obs)

            # update q_table using Bellman formula
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - alpha) * current_q + alpha * (reward + (gamma * max_future_q))
            # print("max_future_q: ", max_future_q, "current_q: ", current_q, "new_q: ", new_q)
            q_table[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state

            done = terminated or truncated  # if terminated or truncated, then the episode is done

        # lower epsilon on next epoch
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay_value
        else:
            epsilon = epsilon_min
        if not (episode % 1000):
            print("epoch: {}, epsilon: {:05f}".format(episode, epsilon))

    print("final q_table: ")
    print(q_table)
    np.save("q_table", q_table)


def predict():
    """
    Predict the agent
    :return:
    """
    print("**** Predict ****")
    q_table = np.load("q_table.npy")
    env = gym.make('MountainCar-v0', render_mode="human")
    # env_test = gym.make('MountainCar-v0')
    discrete_state = to_discrete_state(env.reset(options={}, seed=42)[0])
    done = False
    while not done:
        action_index = np.argmax(q_table[discrete_state])
        observation, reward, terminated, truncated, info = env.step(action_index)
        discrete_state = to_discrete_state(observation)
        done = terminated or truncated


def main():
    """
    Main function

    Description and details of this challenge are available here: https://gymnasium.farama.org/environments/classic_control/mountain_car/

    """
    init()
    if TRAIN:
        train()

    if PREDICT:
        predict()


if __name__ == "__main__":
    main()

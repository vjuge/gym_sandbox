import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0')
# env = gym.make('MountainCar-v0', render_mode="human")

# choose number of grid points to use
n_grid = 10
# grid over observations[0] and actions[1]
gridspace = {}
# learning rate
alpha = 0.1
# discount factor
gamma = 0.98

# number of episodes of training
epochs = 25000


# define a function that makes it easy to get our discrete indices from an observation
def to_discrete_state(observation):
    """
    returns a tuple of (position, velocity) discrete state
    and a tuple of (position, velocity) indices in the grid

    :argument observation: ndarray of continuous observation state (position, velocity)
    """
    # print("** get_discretized_observation **")
    # print("observation: ", observation)
    # print("gridspace: ", gridspace)
    # for (ii, grid) in gridspace.items():
    #     print("ii: ", ii)
    #     print("grid: ", grid)
    #     print("observation[ii]: ", observation[ii])
    #     print("np.abs(grid - observation[ii]): ", np.abs(grid - observation[ii]))
    #     print("np.argmin(np.abs(grid - observation[ii])): ", np.argmin(np.abs(grid - observation[ii])))
    # return tuple of indices in the grid
    index = np.array([np.argmin(np.abs(grid - observation[ii])) for (ii, grid) in gridspace.items()])
    # print("index: ", index)
    state = gridspace[0][index[0]], gridspace[1][index[1]]
    # print("state: ", state)
    return state, index


def explore_or_exploit(_epsilon, _q_table, _index):
    """
    Returns an action according to an epsilon-greedy approach
    :param _epsilon: 
    :param _q_table: 
    :param _index:
    :return: 
    """
    if np.random.random() > _epsilon:
        # exploit
        strategy = "exploit"
        action = np.argmax(_q_table[_index[0], _index[1]])
    else:
        # explore
        strategy = "explore"
        action = np.random.randint(0, env.action_space.n)
    return action, strategy


def main():
    print("**** Initialization ****")
    print("action_space: ", env.action_space)
    # print("action_space_dim: ", env.action_space.n)

    # observation space
    print("observation_space: ", env.observation_space)
    # initial state observation
    obs = env.reset()
    print("initial observation_position: ", obs[0][0])
    print("initial observation_velocity: ", obs[0][1])

    # build grid space = discretize the observation space
    for ii, s in enumerate(zip(env.observation_space.low, env.observation_space.high)):
        gridspace[ii] = np.round(np.linspace(s[0], s[1], n_grid), 2)
    print("grid space positions: ", gridspace[0])
    print("grid space velocity: ", gridspace[1])

    # discretize initial observation to get the initial state
    state, index = to_discrete_state(obs[0])
    print("initial state position:", state[0])
    print("initial state velocity:", state[1])

    # build q_table = initialize the q_table with zeros
    q_table = np.random.uniform(low= -1, high=1, size=(n_grid, n_grid, env.action_space.n))
    # q_table = np.zeros([n_grid, n_grid, env.action_space.n])
    # print("q_table shape:", q_table.shape)
    # print("q_table: ", q_table)

    # exploration/exploitation rate : 1 = 100% exploration, 0 = 100% exploitation
    # 'epsilon greedy' strategy : epsilon is very high at the beginning, then decreases over time
    epsilon = 1.

    epsilon_min = 0.01
    start_epsilon_decaying = 1000
    end_epsilon_decaying = epochs // 2
    epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

    print("**** Training ****")
    for episode in range(epochs):
        # print("episode: ", episode, "epsilon: ", epsilon)
        done = False
        while not done:
            action, strategy = explore_or_exploit(epsilon, q_table, index)
            # print("action: ", action, "strategy: ", strategy)

            obs, reward, terminated, truncated, info = env.step(action)

            if obs[0] >= env.goal_position:
                reward = 1
            # print("obs: ", obs)

            state, index = to_discrete_state(obs)
            # print("state: ", state, "reward: ", reward)
            # print("index: ", index)
            # print("q_table[state]: ", q_table[index[0]][index[1]][action])

            # update q_table using Bellman formula
            max_future_q = np.max(q_table[index[0]][index[1]])
            current_q = q_table[index[0], index[1], action]
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_table[index[0], index[1], action] = new_q

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
    # np.save("q_table", q_table)

    print("**** Predict ****")
    env_test = gym.make('MountainCar-v0', render_mode="human")
    # env_test = gym.make('MountainCar-v0')
    state = env_test.reset(options={})
    state, index = to_discrete_state(state[0])
    done = False
    while not done:
        # print("q_table[index[0], index[1]]", q_table[index[0], index[1]])
        a_index = np.argmax(q_table[index[0], index[1]])
        # print("q_table[{}], position: {:05f}, velocity: {:05f}, action: {}"
        #       .format(q_table[index[0], index[1]], state[0], state[1], a_index))
        state, reward, terminated, truncated, info = env_test.step(a_index)
        state, index = to_discrete_state(state)
        done = terminated or truncated


if __name__ == "__main__":
    main()

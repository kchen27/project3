import gymnasium as gym
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True,)

# Your code for Q2.2 which Executes Random Policy until 1000 episodes
observation,info=env.reset()

reward_table = dict()
transition_table = dict()
transition_table_norm = dict()

for _ in range(1000):
    action=env.action_space.sample()
    prev_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)

    award = reward
    if reward == 1:
        award *= 80
    reward_table[(prev_observation, observation)] = award

    if (prev_observation, observation) not in transition_table:
        transition_table[(prev_observation, observation)] = 1
    else:
        transition_table[(prev_observation, observation)] = transition_table[(prev_observation, observation)] + 1

    if prev_observation not in transition_table_norm:
        transition_table_norm[prev_observation] = 1
    else:
        transition_table_norm[prev_observation] = transition_table_norm[prev_observation] + 1


    if terminated or truncated:
        if reward == 0.0:
            reward_table[(prev_observation, observation)] = -1.0
        observation, info = env.reset()

env.close()


for item in transition_table:
    s = item[0]
    s_p = item[1] # s prime
    transition_table[item] = transition_table[item] / transition_table_norm[s]


# for elem in transition_table:
#     print(elem, transition_table[elem])
#     print(elem, reward_table[elem])


# Your code for Q2.3 which implements Value Iteration

# Makes the reward and transition table easier to look up when doing value iteration
value_table = dict()
possible_transitions = dict()
for item in reward_table:
    key = item[0]
    value_table[key] = 0
    if key not in possible_transitions:
        possible_transitions[key] = [item]
    else:
        possible_transitions[key] = possible_transitions[key] + [item]







gamma = 0.9

value_action_table = dict()
for iteration in range(1000):
    for s in value_table:
        possible_states = possible_transitions[s]
        value_action = []
        for state in possible_states:
            reward = reward_table[state]
            transition = transition_table[state]
            if reward != 0.0:
                value_s = reward
            else:
                value_future = value_table[state[1]]
                value_s = transition * (reward + gamma * value_future)
            value_action.append((value_s, state[1]))

        value_action_table[s] = value_action
        expected_value = sum(val[0] for val in value_action)

        value_table[s] = expected_value

#Your code for Q2.4 which implements policy extraction



optimal_action_table = dict()

for state in value_action_table:
    value = value_action_table[state]

    value.sort()

    optimal_action = value[-1][1]
    if state == optimal_action:
        optimal_action = value[-2][1]

    # print(state, optimal_action, value)
    if state == optimal_action - 4: # Want to move down
        optimal_action_table[state] = 1
    if state == optimal_action + 4: # Move up
        optimal_action_table[state] = 3
    if state == optimal_action - 1: # Move left
        optimal_action_table[state] = 2
    if state == optimal_action + 1: # Move right
        optimal_action_table[state] = 0



#Your code for Q2.5 which execute the optimal policy




env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human", is_slippery=False,)
observation,info=env.reset()


for _ in range(200):
    action=optimal_action_table[observation]
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()


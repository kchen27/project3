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
        award *= 1
    reward_table[(prev_observation, action, observation)] = award

    if (prev_observation, action, observation) not in transition_table:
        transition_table[(prev_observation, action, observation)] = 1
    else:
        transition_table[(prev_observation, action, observation)] = transition_table[(prev_observation, action, observation)] + 1

    if (prev_observation, action) not in transition_table_norm:
        transition_table_norm[(prev_observation, action)] = 1
    else:
        transition_table_norm[(prev_observation, action)] = transition_table_norm[(prev_observation, action)] + 1


    if terminated or truncated:
        if reward == 0.0:
            reward_table[(prev_observation, action, observation)] = -1.0
        observation, info = env.reset()

env.close()


# normalize all the probabilities
for item in transition_table:
    s = item[0]
    a = item[1]
    s_p = item[2] # s prime
    transition_table[item] = transition_table[item] / transition_table_norm[(s, a)]






# Your code for Q2.3 which implements Value Iteration

# Makes the reward and transition table easier to look up when doing value iteration
value_table = dict()
possible_transitions = dict()
for item in reward_table:
    key = item[0]
    action = item[1]
    value_table[key] = 0
    if key not in possible_transitions:
        action_table = {action: [item]}
        possible_transitions[key] = action_table
    else:
        action_table = possible_transitions[key]
        if action not in action_table:
            action_table[action] = [item]
        else:
            action_table[action] = action_table[action] + [item]
        possible_transitions[key] = action_table







gamma = 0.9

Q_value_table = dict()
for iteration in range(1000):
    for s in value_table:
        possible_actions = possible_transitions[s]
        value_action = []
        for action in possible_actions: #i terating through each action
            summation_val = 0
            for s_a_sp in possible_actions[action]: # for finding V_k
                # sp is s prime
                sp = s_a_sp[2]
                reward = reward_table[s_a_sp]
                transition = transition_table[s_a_sp]

                if reward != 0.0:
                    value_s = reward
                else:
                    value_future = value_table[sp]
                    value_s = transition * (reward + gamma * value_future)
                summation_val += value_s
            value_action.append((summation_val, action))
        value_table[s] = max(value_action)[0]
        Q_value_table[s] = value_action


# Gets the optimal values from the Q* table
optimal_values = dict()
for item in Q_value_table:
    optimal_values[item] = max(Q_value_table[item])[0]



# #Your code for Q2.4 which implements policy extraction



optimal_action_table = dict()

for state in Q_value_table:
    value = Q_value_table[state]
    optimal_action = max(value)[1]
    optimal_action_table[state] = optimal_action






# #Your code for Q2.5 which execute the optimal policy




env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human", is_slippery=True,)
observation,info=env.reset()


for _ in range(200):
    action=optimal_action_table[observation]
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()


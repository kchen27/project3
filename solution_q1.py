import gymnasium as gym
env = gym.make('Blackjack-v1',natural=False,sab=False)





observation, info = env.reset()



alpha = 0.001
discount = 0.95 # Gamma


qsa_table = dict()# key is a tuple (int, int) being (observation, possible action)









print("Q Learning in Progress...")

for n in range(100000):
    action = env.action_space.sample() # agent policy that uses the observation and info

    if (observation, action) not in qsa_table:
        qsa_table[(observation, action)] = 0

    prev_observation = observation


    observation, reward, terminated, truncated, info = env.step(action)
    if (observation, 0) not in qsa_table:
        qsa_table[(observation, 0)] = 0
    if (observation, 1) not in qsa_table:
        qsa_table[(observation, 1)] = 0



    option1 = qsa_table[(observation, 0)] # We want the max of the two actions
    option2 = qsa_table[(observation, 1)]

    difference = (reward * 100) + discount * (max(option1, option2)) - qsa_table[(prev_observation, action)]
    qsa_table[(prev_observation, action)] = qsa_table[(prev_observation, action)] + alpha * difference



    if terminated or truncated:
        observation, info = env.reset()


env.close()
print("Q Learning Done")






print("Testing Q Learning Table")


observation, info = env.reset()
wins = 0
losses = 0
for _ in range(4000):
    choice = max((qsa_table[(observation, 0)], 0), (qsa_table[(observation, 1)], 1))
    action = choice[1]





    observation, reward, terminated, truncated, info = env.step(action)





    if terminated or truncated:
        if reward == 1:
            wins += 1
        else:
            losses += 1
        observation, info = env.reset()


env.close()
print("Wins:", wins, "Losses:", losses)
print("Winrate:", wins/(wins + losses))

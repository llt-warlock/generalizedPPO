import envs
from envs.gym_wrapper import GymWrapper, VecEnv

# env = GymWrapper("env_0")
#
# # Reset the environment before starting
# observation = env.reset()
#
# # Run for a fixed number of steps or until the episode ends
# done = False
# counter = 0
# while counter < 5:
#     # Choose a random action
#     action = env.action_space.sample()
#     #print("action : ", action)
#
#     # Take a step in the environment and receive the new observation, reward, and done flag
#     observation, reward, done, info = env.step(action)
#     #print("observation : ", observation, " reward : ", reward, " done : ", done, " info : ", info)
#
#     # Optionally print out the observation to make sure it's what you expect
#     #print(observation)
#
#     counter += 1

vec_env = VecEnv("env_0", 3)
print(vec_env.env_id)
print(vec_env.action_space)
print(vec_env.observation_space)

print(int(9e6/12))
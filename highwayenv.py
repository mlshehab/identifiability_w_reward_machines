import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
# %matplotlib inline
import pprint
import sys
# sys.path.append(r'C:\Users\mlshehab\Desktop\finite-mdp')

env = gym.make("intersection-v0",render_mode="rgb_array")
env.reset()

a = env.unwrapped.to_finite_mdp()
print(a.transition)
for _ in range(10):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    print(f"The action is: {action}")
    action = 2
    obs, reward, done, truncated, info = env.step(action)
    env.render()
pprint.pprint(env.unwrapped.config)
plt.imshow(env.render())
plt.show()
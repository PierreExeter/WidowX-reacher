# calc reward manually
import numpy as np


# reward:  -0.03629627313189004


# info:  {
# 'total_distance': 0.19051580808922403, 
# 'goal position': array([0.14, 0.  , 0.26]), 
# 'tip position': array([-0.01628839,  0.00086599,  0.36894706]), 
# 'joint position': array([-0.05311593, -1.30008376,  0.95553899, -1.45568061, -0.04549807, 0. ])
# }


end = np.array([-0.01628839,  0.00086599,  0.36894706])
goal = np.array([0.14, 0.  , 0.26])

print(end)
print(goal)

reward1 = - (np.linalg.norm(end - goal) ** 2)
print(reward1)

dist = np.sqrt(-reward1)
print(dist)

reward2 = -((end[0]-goal[0])**2 + (end[1]-goal[1])**2 + (end[2]-goal[2])**2)
print(reward2)
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

count = 0

while True:

    # import as pkl to be able to read list as list (not string)
    # df = pd.read_csv("res_episode_0.csv")
    
    # keep trying to read the pickle until success (sometimes it is not complete)
    while True:
        try:
            df = pd.read_pickle("res_episode_0.pkl")
        except EOFError:
            continue
        break

    x_tip_list = []
    y_tip_list = []
    z_tip_list = []

    for index, row in df.iterrows():
        tip = row['tip position']
        goal = row['goal position']
        # x_tip_list.append(tip[0])
        # y_tip_list.append(tip[1])
        # z_tip_list.append(tip[2])

        # ax.plot(x_tip_list, y_tip_list, zs=z_tip_list, marker='o', color='b', markersize=1)
        ax.plot([tip[0]], [tip[1]], zs=[tip[2]], marker='o', color='b', markersize=1)

        # only print the goal once
        if index == 0:
            ax.plot([goal[0]], [goal[1]], zs=[goal[2]], marker='x', color='k', linestyle="None")


    # only plot the legend once
    if count == 0:
        ax.legend(["tip position", "goal position"], loc="lower left")
    
    plt.pause(0.01)
    print(count)
    count += 1
    
    
    # plt.show()

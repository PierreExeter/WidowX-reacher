import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from ast import literal_eval

df1 = pd.read_csv("res_episode_0.csv")
df1.info()


# df = pd.read_csv("res_episode_0.csv", converters={"goal position": literal_eval})
# df.info()

df2 = pd.read_pickle("res_episode_0.pkl")
df2.info()
# df2.dtypes

print(df1)
print(df2)



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax1 = fig.add_subplot(111,projection='3d')

# for index, row in df.iterrows():
#     # print(row['tip position'], row['goal position'])
#     tip = row['tip position']
#     goal = row['goal position']

#     # print(tip.type)

#     # ax.plot(tip[0], tip[1], tip[2], label='parametric curve')



# ax.legend()
# plt.show()



# # for tip_pos in df['tip position']:
# #     print(tip_pos)
# # print(df)

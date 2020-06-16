import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# plot best value from optimisation report

log_dir = "logs/opti100t_0.5M_widowx_reacher-v5/"

algo_list = []
best_list = []


file_path = log_dir+"a2c/"
report_name = "report_widowx_reacher-v5_100-trials-500000-tpe-median_1592156839.csv"
algo_list.append("a2c")
df = pd.read_csv(file_path+report_name)
df = df.loc[df['state'] == 'COMPLETE']   # filter
best_list.append(-df['value'].min())

file_path = log_dir+"acktr/"
report_name = "report_widowx_reacher-v5_100-trials-500000-tpe-median_1592171741.csv"
algo_list.append("acktr")
df = pd.read_csv(file_path+report_name)
df = df.loc[df['state'] == 'COMPLETE']   # filter
best_list.append(-df['value'].min())

file_path = log_dir+"ppo2/"
report_name = "report_widowx_reacher-v5_100-trials-500000-tpe-median_1592312653.csv"
algo_list.append("ppo2")
df = pd.read_csv(file_path+report_name)
df = df.loc[df['state'] == 'COMPLETE']   # filter
best_list.append(-df['value'].min())

# file_path = log_dir+"td3/"
# report_name = "report_ReachingJaco-v1_10-trials-100000-tpe-median_1588497301.csv"
# algo_list.append("td3")
# df = pd.read_csv(file_path+report_name)
# df = df.loc[df['state'] == 'COMPLETE']   # filter
# best_list.append(-df['value'].min())


print(algo_list)
print(best_list)

plt.figure()
plt.bar(algo_list, best_list)
plt.ylabel("reward")
plt.xlabel("algo")
plt.title("Algo comparison")
# plt.show()
plt.savefig(log_dir+"opti_comparison.png", dpi=100)



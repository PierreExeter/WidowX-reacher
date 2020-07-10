from pathlib import Path
import pandas as pd
import numpy as np


algo_list = ["a2c", "acktr", "ddpg", "ppo2", "sac", "td3", "trpo", "her_sac", "her_td3"]


for algo in algo_list:
    print("***********"+algo+"********")

    log_dir = "logs/train_0.2M_widowx_reacher-v7_SONIC/"+algo

    res_file_list = []
    reward_list = []

    for path in Path(log_dir).rglob('stats.csv'):
        # print(path)
        df = pd.read_csv(path)
        reward = df["Eval mean reward"][0]

        res_file_list.append([path, reward])


    X = np.array(res_file_list)
    print(X[:, 1])

    maxid = np.argmax(X[:, 1], axis=0)
    print(maxid)
    print("best seed for "+ algo +" is: ", X[maxid, :])


import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob


file_path = "logs/opti100t_0.5M_widowx_reacher-v5/ppo2/"
#report_name = "report_ReachingJaco-v1_20-trials-10000-tpe-median_1588163765.csv"

report_name = glob.glob(file_path+'*csv')[0]
print(report_name)


df = pd.read_csv(report_name)

print(df)

# drop trials that are not complete (to check if it works)
df.drop(df[df['state'] != 'COMPLETE'].index, inplace=True)

# remove columns that I don't want to plot
df.drop(['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], axis=1, inplace=True)

cols = df.columns
print(cols)


for col in cols[1:]:

    print(col)

    df.plot.scatter(x=col, y='value')
    plt.savefig(file_path+col+"_vs_value.png", dpi=100)
    # plt.show()
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



file_path = "logs/opti20t_0.01M_ReachingJaco-v1/a2c/"
report_name = "report_ReachingJaco-v1_20-trials-10000-tpe-median_1588163765.csv"


df = pd.read_csv(file_path+report_name)

print(df)

cols = df.columns
print(cols)

for col in cols[5:-2]:
    print(col)

    df.plot.scatter(x=col, y='value')
    plt.savefig(file_path+col+"_vs_value.png", dpi=100)
    # plt.show()
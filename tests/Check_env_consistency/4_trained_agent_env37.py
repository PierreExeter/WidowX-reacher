import pandas as pd
import matplotlib.pyplot as plt
from plot_lib import plot_df_1joint_env37

filename = "4_trained_agent_env37"

log_df = pd.read_csv("logs/"+filename+".csv")
print(log_df)

plot_df_1joint_env37(log_df, "plots/"+filename+".png")

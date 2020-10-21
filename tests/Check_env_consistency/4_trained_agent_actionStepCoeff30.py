import pandas as pd
import matplotlib.pyplot as plt
from plot_lib import plot_df

filename = "4_trained_agent_actionStepCoeff30"

log_df = pd.read_csv("logs/"+filename+".csv")
print(log_df)

plot_df(log_df, "plots/"+filename+".png")

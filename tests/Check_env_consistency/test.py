import matplotlib.pyplot as plt
import pandas as pd
from plot_lib import plot_df


log_df = pd.read_csv("logs/1_action0.csv")
print(log_df)


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

# p1, = log_df.plot(x='timestep', y='dist', ax=host, color="b", marker="x")
# p2, = log_df.plot(x='timestep', y='est_vel', ax=par1, color="r", marker="+")
# p3, = log_df.plot(x='timestep', y='est_acc', ax=par2, color="g", marker="o")

p1, = host.plot(log_df['timestep'], log_df['dist'], "bx", label="dist")
p2, = host.plot(log_df['timestep'], log_df['est_vel'], "r+", label="est_vel")
p3, = host.plot(log_df['timestep'], log_df['est_acc'], "g*", label="est_acc")

# host.set_xlim(0, 2)
# host.set_ylim(0, 2)
# par1.set_ylim(0, 4)
# par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines])

plt.show()

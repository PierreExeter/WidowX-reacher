import pandas as pd
import matplotlib.pyplot as plt


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_df(log_df, filename):
    """ plot episode logs stored in the dataframe log_df and save it under filename """

    fig, axs = plt.subplots(4, 4, figsize=(20, 10), dpi=300, sharex=True)

    log_df.plot(x='timestep', y='current_joint_pos_1', ax=axs[0, 0])
    log_df.plot(x='timestep', y='new_joint_pos_1', ax=axs[0, 0])
    log_df.plot(x='timestep', y='joint1_min', ax=axs[0, 0], style="r--")
    log_df.plot(x='timestep', y='joint1_max', ax=axs[0, 0], style="r--")

    log_df.plot(x='timestep', y='current_joint_pos_2', ax=axs[1, 0])
    log_df.plot(x='timestep', y='new_joint_pos_2', ax=axs[1, 0])
    log_df.plot(x='timestep', y='joint2_min', ax=axs[1, 0], style="r--")
    log_df.plot(x='timestep', y='joint2_max', ax=axs[1, 0], style="r--")

    log_df.plot(x='timestep', y='current_joint_pos_3', ax=axs[2, 0])
    log_df.plot(x='timestep', y='new_joint_pos_3', ax=axs[2, 0])
    log_df.plot(x='timestep', y='joint3_min', ax=axs[2, 0], style="r--")
    log_df.plot(x='timestep', y='joint3_max', ax=axs[2, 0], style="r--")

    log_df.plot(x='timestep', y='current_joint_pos_4', ax=axs[3, 0])
    log_df.plot(x='timestep', y='new_joint_pos_4', ax=axs[3, 0])
    log_df.plot(x='timestep', y='joint4_min', ax=axs[3, 0], style="r--")
    log_df.plot(x='timestep', y='joint4_max', ax=axs[3, 0], style="r--")

    log_df.plot(x='timestep', y='current_joint_pos_5', ax=axs[0, 2])
    log_df.plot(x='timestep', y='new_joint_pos_5', ax=axs[0, 2])
    log_df.plot(x='timestep', y='joint5_min', ax=axs[0, 2], style="r--")
    log_df.plot(x='timestep', y='joint5_max', ax=axs[0, 2], style="r--")

    log_df.plot(x='timestep', y='current_joint_pos_6', ax=axs[1, 2])
    log_df.plot(x='timestep', y='new_joint_pos_6', ax=axs[1, 2])
    log_df.plot(x='timestep', y='joint6_min', ax=axs[1, 2], style="r--")
    log_df.plot(x='timestep', y='joint6_max', ax=axs[1, 2], style="r--")

    log_df.plot(x='timestep', y='action_1', ax=axs[0, 1])
    log_df.plot(x='timestep', y='action_low1', ax=axs[0, 1], style="r--")
    log_df.plot(x='timestep', y='action_high1', ax=axs[0, 1], style="r--")

    log_df.plot(x='timestep', y='action_2', ax=axs[1, 1])
    log_df.plot(x='timestep', y='action_low2', ax=axs[1, 1], style="r--")
    log_df.plot(x='timestep', y='action_high2', ax=axs[1, 1], style="r--")

    log_df.plot(x='timestep', y='action_3', ax=axs[2, 1])
    log_df.plot(x='timestep', y='action_low3', ax=axs[2, 1], style="r--")
    log_df.plot(x='timestep', y='action_high3', ax=axs[2, 1], style="r--")

    log_df.plot(x='timestep', y='action_4', ax=axs[3, 1])
    log_df.plot(x='timestep', y='action_low4', ax=axs[3, 1], style="r--")
    log_df.plot(x='timestep', y='action_high4', ax=axs[3, 1], style="r--")

    log_df.plot(x='timestep', y='action_5', ax=axs[0, 3])
    log_df.plot(x='timestep', y='action_low5', ax=axs[0, 3], style="r--")
    log_df.plot(x='timestep', y='action_high5', ax=axs[0, 3], style="r--")

    log_df.plot(x='timestep', y='action_6', ax=axs[1, 3])
    log_df.plot(x='timestep', y='action_low6', ax=axs[1, 3], style="r--")
    log_df.plot(x='timestep', y='action_high6', ax=axs[1, 3], style="r--")

    lns11 = log_df.plot(x='timestep', y='reward', ax=axs[2, 2], color="b")
    ax22 = axs[2, 2].twinx()
    lns22 = log_df.plot(x='timestep', y='return', ax=ax22, color="r")

    lns1 = log_df.plot(x='timestep', y='dist', ax=axs[2, 3], color="b", marker="x")
    ax2 = axs[2, 3].twinx()
    lns2 = log_df.plot(x='timestep', y='est_vel', ax=ax2, color="r", marker="+")

    log_df.plot(x='timestep', y='est_acc', ax=axs[3, 2], color="g", marker="*")

    log_df.plot(x='timestep', y='target_x', ax=axs[3, 3], style='or')
    log_df.plot(x='timestep', y='target_y', ax=axs[3, 3], style='ob')
    log_df.plot(x='timestep', y='target_z', ax=axs[3, 3], style='og')
    log_df.plot(x='timestep', y='tip_x', ax=axs[3, 3], style='xr')
    log_df.plot(x='timestep', y='tip_y', ax=axs[3, 3], style='xb')
    log_df.plot(x='timestep', y='tip_z', ax=axs[3, 3], style='xg')

    axs[0, 0].set_ylabel("rad")
    axs[1, 0].set_ylabel("rad")
    axs[2, 0].set_ylabel("rad")
    axs[3, 0].set_ylabel("rad")
    axs[0, 2].set_ylabel("rad")
    axs[1, 2].set_ylabel("rad")

    axs[0, 1].set_ylabel("rad")
    axs[1, 1].set_ylabel("rad")
    axs[2, 1].set_ylabel("rad")
    axs[3, 1].set_ylabel("rad")
    axs[0, 3].set_ylabel("rad")
    axs[1, 3].set_ylabel("rad")

    axs[2, 2].set_ylabel("m^2")

    axs[2, 2].set_ylabel("reward (m**2)", color="b")
    ax22.set_ylabel("return (m**2)", color="r")
    axs[2, 2].tick_params(axis='y', labelcolor="b")
    ax22.tick_params(axis='y', labelcolor="r")

    axs[2, 3].set_ylabel("m^2")

    axs[2, 3].set_ylabel("dist (m)", color="b")
    ax2.set_ylabel("vel (m/s)", color="r")
    axs[2, 3].tick_params(axis='y', labelcolor="b")
    ax2.tick_params(axis='y', labelcolor="r")
    axs[2, 3].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axs[3, 2].set_ylabel("acc (m/s**2)")

    axs[3, 3].set_ylabel("coordinates (m)")

    axs[3, 3].legend(loc="upper right")
    # ax3.legend(bbox_to_anchor=(1, 1.05))
    # ax4.legend(bbox_to_anchor=(1.2, 1.05))

    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)

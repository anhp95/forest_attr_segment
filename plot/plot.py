# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

dir_ = r"D:\Publication\IGARSS 2021\tree_spec\material"
csv_dir = "loss_acc_performance"
figure_dir = "figure"

#%%
attr = "spec"
csv_file = os.path.join(dir_, csv_dir, f"{attr}.csv")
fig_file = os.path.join(dir_, figure_dir, f"{attr}_acc.png")

df = pd.read_csv(csv_file)

epoch = df.epoch.values + 1

acc_2d_p2 = df.acc_2d_p2
acc_2d_p1p2 = df.acc_2d_p1p2
acc_2d_p1p2p3 = df.acc_2d_p1p2p3
acc_3d_org = df.acc_3d_org_p1p2p3
acc_3d_adj = df.acc_3d_adj_p1p2p3

fig, ax1 = plt.subplots()

ax1.plot(epoch, acc_2d_p2, "pink", label="2D UNET - P2")
ax1.plot(epoch, acc_2d_p1p2, "g-", label="2D UNET - P1P2")
ax1.plot(epoch, acc_2d_p1p2p3, "y-", label="2D UNET - P1P2P3")
ax1.plot(epoch, acc_3d_org, "r-", label="3D UNET - P1P2P3")
ax1.plot(epoch, acc_3d_adj, "b-", label="Our model")

ax1.set_xlabel("Number of epochs")
ax1.set_ylabel("Overall Accuracy")

plt.legend()
plt.show()

# fig.savefig(fig_file, format="png", dpi=1200)

#%%
spec_file = os.path.join(dir_, csv_dir, "spec.csv")
fig_file = os.path.join(dir_, figure_dir, "spec_acc_new.png")
df = pd.read_csv(spec_file)

epoch = df.epoch.values + 1

acc_2d_p2 = df.acc_2d_p2
acc_2d_p1p2 = df.acc_2d_p1p2
acc_2d_p1p2p3 = df.acc_2d_p1p2p3
acc_3d_p1p2p3 = df.loss_3d_p1p2p3
acc_new = df.loss_new

fig, ax1 = plt.subplots()


ax1.plot(epoch, acc_3d_p1p2p3, "b-", label="Old model")
ax1.plot(epoch, acc_new, "g-", label="Improved model")

ax1.set_xlabel("Number of epochs")
ax1.set_ylabel("Loss")

plt.legend()
plt.show()

fig.savefig(fig_file, format="png", dpi=1200)

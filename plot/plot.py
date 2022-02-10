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
age_file = os.path.join(dir_, csv_dir, "age.csv")
fig_file = os.path.join(dir_, figure_dir, "age_acc.png")

df = pd.read_csv(age_file)

epoch = df.epoch.values + 1

acc_2d_p2 = df.acc_2d_p2
acc_2d_p1p2 = df.acc_2d_p1p2
acc_2d_p1p2p3 = df.acc_2d_p1p2p3
acc_3d_p1p2p3 = df.acc_3d_p1p2p3

fig, ax1 = plt.subplots()

ax1.plot(epoch, acc_2d_p2, "r-", label="UNET (P2)")
ax1.plot(epoch, acc_2d_p1p2, "g-", label="UNET (P1 + P2)")
ax1.plot(epoch, acc_2d_p1p2p3, "y-", label="UNET (P1 + P2 + P3)")
ax1.plot(epoch, acc_3d_p1p2p3, "b-", label="Our model (P1 + P2 + P3)")

ax1.set_xlabel("Number of epochs")
ax1.set_ylabel("Overall Accuracy")

plt.legend()
plt.show()

fig.savefig(fig_file, format="png", dpi=1200)

#%%
spec_file = os.path.join(dir_, csv_dir, "spec.csv")
fig_file = os.path.join(dir_, figure_dir, "spec_acc.png")
df = pd.read_csv(spec_file)

epoch = df.epoch.values + 1

acc_2d_p2 = df.acc_2d_p2
acc_2d_p1p2 = df.acc_2d_p1p2
acc_2d_p1p2p3 = df.acc_2d_p1p2p3
acc_3d_p1p2p3 = df.acc_3d_p1p2p3

fig, ax1 = plt.subplots()

ax1.plot(epoch, acc_2d_p2, "r-", label="UNET (P2)")
ax1.plot(epoch, acc_2d_p1p2, "g-", label="UNET (P1 + P2)")
ax1.plot(epoch, acc_2d_p1p2p3, "y-", label="UNET (P1 + P2 + P3)")
ax1.plot(epoch, acc_3d_p1p2p3, "b-", label="Our model (P1 + P2 + P3)")

ax1.set_xlabel("Number of epochs")
ax1.set_ylabel("Overall Accuracy")

plt.legend()
plt.show()

fig.savefig(fig_file, format="png", dpi=1200)
# %%

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

if __name__ == "__main__":
    # read log file
    root_dir = "C:\\Users\\xinyi\\Documents"
    # ckpt_path = os.path.join(root_dir, "Checkpoints", "try_SR_")
    # ckpt_path = os.path.join(root_dir, "Checkpoints", "try0")
    ckpt_path = os.path.join(root_dir, "Checkpoints", "try_38")
    # acc_path = os.path.join(ckpt_path, "acc.csv")
    log_path = os.path.join(ckpt_path, "log.csv")

    # create plot design
    colors = [[ 78.0/255.0,121.0/255.0,167.0/255.0], # 0_blue
              [255.0/255.0, 87.0/255.0, 89.0/255.0], # 1_red
              [ 89.0/255.0,169.0/255.0, 79.0/255.0], # 2_green
              [237.0/255.0,201.0/255.0, 72.0/255.0], # 3_yellow
              [242.0/255.0,142.0/255.0, 43.0/255.0], # 4_orange
              [176.0/255.0,122.0/255.0,161.0/255.0], # 5_purple
              [255.0/255.0,157.0/255.0,167.0/255.0], # 6_pink 
              [118.0/255.0,183.0/255.0,178.0/255.0], # 7_cyan
              [156.0/255.0,117.0/255.0, 95.0/255.0], # 8_brown
              [186.0/255.0,176.0/255.0,172.0/255.0]] # 9_gray

    # create figure
    ax = plt.axes()
    ax.set_facecolor("white")
    # plt.grid(True, linestyle='-', color=[0.8,0.8,0.8])
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('#000000')
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['mathtext.default']='regular'
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    # load and plot loss
    df = pd.read_csv(log_path)
    # df.insert(0, "epoch", list(range(100)), True)
    # print(df)
    # df = df_all[df_all['epoch'] <= 12]

    sns.lineplot(x="epoch", y="train_loss", data=df, label="Train Loss", color=colors[0], linewidth=2)
    sns.lineplot(x="epoch", y="test_loss", data=df,  label="Test Loss", color=colors[2], linewidth=2)
    # plt.ylim((0, 1))
    plt.ylabel('Training loss')
    plt.xlabel('Number of epochs')   
    legend =plt.legend(loc='upper right', fontsize=14)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()

    # create figure
    # ax = plt.axes()
    # ax.set_facecolor("white")
    # # plt.grid(True, linestyle='-', color=[0.8,0.8,0.8])
    # ax = plt.gca()
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_color('#000000')

    # plt.rcParams.update({'font.size': 18})
    # plt.rcParams['mathtext.default']='regular'
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42

    # N = 2
    # menMeans = (21, 18)
    # womenMeans = (2, 2)
    # new_add = (2,5)
    # ind = ('Sim', 'Real')    # the x locations for the groups
    # width = 0.4    # the width of the bars: can also be len(x) sequence

    # p1 = plt.bar(ind, menMeans, width, color=colors[2])
    # p2 = plt.bar(ind, womenMeans, width, color=colors[3], bottom=menMeans)
    # p3 = plt.bar(ind, new_add, width, color=colors[1], bottom=(23,20))
    
    # # load and plot loss
    # df_all = pd.read_csv(acc_path)
    # df = df_all[df_all['iteration'] <= 11]

    # sns.lineplot(x="iteration", y="pick_data", data=df,  label="Pick Data", color=colors[0], linewidth=2)
    # sns.lineplot(x="iteration", y="sep_data", data=df, label="Sep Data", color=colors[1], linewidth=2)

    # sns.lineplot(x="iteration", y="all_data", data=df,  label="All Data", color=colors[2], linewidth=3)
    # plt.ylim((0, 1.1))
    # plt.ylabel('Val Accuract')
    # legend =plt.legend(['Success','Not Optimal','Failure'],loc='lower center', fontsize=14)
    # legend =plt.legend(loc='lower right', fontsize=14)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.show()

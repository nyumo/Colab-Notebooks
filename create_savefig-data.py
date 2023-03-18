#!/usr/bin/env python
# coding: utf-8
#作成日: 2021-11-28
#改良: 2021-11-30

# In[1]:
import platform
import datetime
import numpy as np

#file形式
import csv
import pandas as pd

#グラフ描画
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os #システム操作系
from pathlib import Path #ファイル操作系

print('python_version: ',platform.python_version())#python_ver表示

dt_now = datetime.datetime.now()#実行時刻表示
print('現在時刻：', dt_now)


# In[2]:


# グラフの初期設定
plt.rcParams["figure.figsize"] = [3.14, 3.14] # 図の縦横のサイズ([横(inch),縦(inch)])
plt.rcParams["figure.dpi"] = 200 # dpi(dots per inch)
plt.rcParams["figure.facecolor"] = 'white' # 図の背景色
plt.rcParams["figure.edgecolor"] = 'black' # 枠線の色
plt.rcParams["font.family"] = "serif"       # 使用するフォント
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 14              # 基本となるフォントの大きさ

plt.rcParams["xtick.direction"] = "in"      # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["ytick.direction"] = "in"      # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["xtick.bottom"] = True         # 下部に目盛り線を描くかどうか
plt.rcParams["ytick.left"] = True           # 左部に目盛り線を描くかどうか
plt.rcParams["xtick.major.size"] = 5.0      # x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 5.0      # y軸主目盛り線の長さ
plt.rcParams["xtick.major.width"] = 0.3     # x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 0.3     # y軸主目盛り線の線幅
plt.rcParams["xtick.minor.visible"] = False # x軸副目盛り線を描くかどうか
plt.rcParams["ytick.minor.visible"] = False # y軸副目盛り線を描くかどうか
plt.rcParams["xtick.minor.size"] = 5.0      # x軸副目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5.0      # y軸副目盛り線の長さ
plt.rcParams["xtick.minor.width"] = 0.3     # x軸副目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 0.3     # y軸副目盛り線の線幅
plt.rcParams["xtick.labelsize"] = 10        # 目盛りのフォントサイズ
plt.rcParams["ytick.labelsize"] = 10        # 目盛りのフォントサイズ
plt.rcParams["xtick.major.pad"] = 10.0      # x軸から目盛までの距離
plt.rcParams["ytick.major.pad"] = 10.0      # y軸から目盛までの距離

plt.rcParams["axes.labelsize"] = 10         # 軸ラベルのフォントサイズ
plt.rcParams["axes.linewidth"] = 1.0        # グラフ囲う線の太さ
plt.rcParams["axes.grid"] = False           # グリッドを表示するかどうか


# In[3]:


# このファイルの存在するフォルダの絶対パスを取得
dir_name = str(Path().resolve())
print('このファイルの存在するフォルダ：', dir_name)

# 保存先フォルダのパス作成
save_folder = os.path.join(dir_name, dt_now.strftime("%Y%m%d"))
print('保存フォルダ：', save_folder)

# 保存先フォルダの作成(既に存在する場合は無視される)
os.makedirs(save_folder, exist_ok=True)

# In[4]:
sample = 'tio2' #試料名

#save_name/fileの指定
save_name = dt_now.strftime("%Y-%m-%d_%H-%M-%S_") +sample

#.csvfile読み込み
data_frame = pd.read_csv('time_P.csv')
#data_frame = pd.read_pickle('2022-01-27_14-15-23_IMtiO2-sus304.pkl')
#data_frame.columns = ['Col_0', 'Col_1']        #横軸のヘッダー指定
#data_frame.index = ['Row_0', 'Row_1', 'Row_2'] #縦軸のヘッダー指定
df = data_frame.values.T


print(data_frame)

print('data_shape: ', df.shape)

x_unit = 'μs' #単位
y_unit = 'GPa' #単位
font_size = 30   #フォントサイズ
x_data = df[0] #横軸データ
y_data = df[1] #縦軸データ
x_lim = [0.0, 15.0] #定義域
y_lim = [0.0, 40.0]  #値域

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#data_frame.plot.scatter(x = 'x', y = 'x^1', color = 'black', ax = ax)   #散布図 x^1
#data_frame.plot.scatter(x = 'x', y = 'x^2', color = 'red', ax = ax)     #散布図 x^2
#data_frame.plot.scatter(x = 'x', y = 'x^3', color = 'blue', ax = ax)    #散布図 x^3
data_frame.plot.scatter(x = 'time', y = 'pn', color = 'red', ax = ax)           #プロット x^1
data_frame.plot.scatter(x = 'timep', y = 'pnp', color = 'blue', ax = ax)             #プロット x^2
#data_frame.plot.step(x = 'x', y = 'x^3', color = 'blue', ax = ax)            #プロット x^3

ax.step(x = df[2], y = df[3],where='post', c = 'r')
ax.step(x = df[0], y = df[1],where='post', c = 'b')

ax.set_xlabel(f'T [{x_unit}]', fontsize=font_size)
ax.set_ylabel(f'P [{y_unit}]', fontsize=font_size)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.legend(frameon=False)
plt.tick_params(labelsize=font_size)
ax.grid(False)
ax.set_aspect(1./ax.get_data_ratio()) # グラフを正方形にする
save_file = os.path.join(save_folder, save_name + '.jpg') # 保存先のファイルパス作成
fig.savefig(save_file, format="jpg", bbox_inches="tight")
plt.tight_layout()
plt.show()


# In[6]:
#graph保存先
print(save_file)

#pickel保存先
save_file = os.path.join(save_folder, save_name + '.pkl') # 保存先のファイルパス作成
data_frame.to_pickle(save_file)
print(save_file)

#csvバックアップ保存
save_file = os.path.join(save_folder, save_name + '.csv') # 保存先のファイルパス作成
data_frame.to_csv(save_file)
print(save_file)


# In[ ]:





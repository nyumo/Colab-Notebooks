#作成日: 2023-01-14
#改良: 2023-01-14

import platform
import datetime
import numpy as np

#file形式
import csv
import pandas as pd

import os #システム操作系
from pathlib import Path #ファイル操作系

print('python_version: ',platform.python_version())#python_ver表示

dt_now = datetime.datetime.now()#実行時刻表示
print('現在時刻：', dt_now)

# このファイルの存在するフォルダの絶対パスを取得
dir_name = str(Path().resolve())
print('このファイルの存在するフォルダ：', dir_name)

# 保存先フォルダのパス作成
save_folder = os.path.join(dir_name, dt_now.strftime("%Y%m%d"))
print('保存フォルダ：', save_folder)

# 保存先フォルダの作成(既に存在する場合は無視される)
os.makedirs(save_folder, exist_ok=True)


save_name = '+1.0deg4000ps'

#.csvfile読み込み
data_frame = pd.read_csv(save_name + '.csv')
df = data_frame.values.T

print(data_frame)
print('data_shape: ', df.shape)

Imax = np.max(data_frame['Y'])
Imin = np.min(data_frame['Y'])

I = (data_frame['Y']-Imin)/(Imax-Imin)
data_frame['Y'] = I

#csv保存
save_file = os.path.join(save_folder, save_name + '_nml.csv') # 保存先のファイルパス作成
data_frame.to_csv(save_file, index=False, header=False)
print(save_file)
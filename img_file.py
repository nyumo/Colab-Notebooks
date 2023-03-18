#!/usr/bin/env python
# coding: utf-8
#作成日: 2022-04-15
#改良日: 2022-06-25

from cProfile import run
from cgi import parse_multipart
import imp
from PIL import Image
import numpy as np
import os #システム操作系
from pathlib import Path #ファイル操作系
import datetime
import pandas as pd

dt_now = datetime.datetime.now()#実行時刻表示
print('現在時刻：', dt_now)

# このファイルの存在するフォルダの絶対パスを取得
dir_name = str(Path().resolve())
print('このファイルの存在するフォルダ：', dir_name)

# 保存先フォルダのパス作成
save_folder = os.path.join(dir_name, 'SACLA2022A_png')
print('保存フォルダ：', save_folder)

# 保存先フォルダの作成(既に存在する場合は無視される)
os.makedirs(save_folder, exist_ok=True)

#取得するランナンバー
block_num = 3
plate_num = 2
start = 1167103
stop = 1167202
run_length = np.array(range(start, stop+1)).size


data_frame = pd.DataFrame({'run_num': np.zeros(run_length),
                    'pre_max': np.zeros(run_length),
                    'p&p_max': np.zeros(run_length),
                    'pre_200': np.zeros(run_length),
                    'p&p_200': np.zeros(run_length),
                    })

arr_img = np.zeros((3840, 3840)) #画像サイズ
for run_i in range(start, stop+1):
    data_path1 = f'C:\\Users\\okuch\\Desktop\\sacla2022A\\{run_i}\\data_000001.img'
    data_path2 = f'C:\\Users\\okuch\\Desktop\\sacla2022A\\{run_i}\\data_000002.img'
    save_name = f'{run_i}'
    num = run_i-start
    data_frame['run_num'][num] = int(run_i)
    
    #pre画像png変換&保存
    save_file = os.path.join(save_folder, save_name + '_pre.png') # 保存先のファイルパス作成
    img = Image.open(data_path1)
    i_img = np.array(img)
    img_200 = i_img[0:500, 500:1000]
    #print(img_200.shape)
    #print(img_200.max())
    img = img.convert('L')
    img.save(save_file)
    print(save_name)
    data_frame['pre_max'][num] = i_img.max()
    data_frame['pre_200'][num] = img_200.max()

    #p&p画像png変換&保存
    save_file = os.path.join(save_folder, save_name + '_p&p.png') # 保存先のファイルパス作成
    img = Image.open(data_path2)
    i_img = np.array(img)
    img_200 = i_img[0:500, 500:1000]
    img = img.convert('L')
    img.save(save_file)
    print(save_name)
    data_frame['p&p_max'][num] = i_img.max()
    data_frame['p&p_200'][num] = img_200.max()

    #csv保存
    save_name = dt_now.strftime("%Y-%m-%d_%H-%M-%S_")
    save_file = os.path.join(save_folder, save_name + f'{block_num}-{plate_num}.csv') # 保存先のファイルパス作成
    data_frame.to_csv(save_file)

print(data_frame)
#csv保存
save_name = 'block'
save_file = os.path.join(save_folder, save_name + f'{block_num}-{plate_num}.csv') # 保存先のファイルパス作成
data_frame.to_csv(save_file)
print(save_file)


"""
#積算
    arr_img = arr_img + i_img

arr_img = arr_img / run_i
image_ave = Image.fromarray(arr_img.astype(np.uint8))
print(image_ave.mode)
save_name = "sekisan"
save_file = os.path.join(save_folder, save_name + '.png') # 保存先のファイルパス作成
image_ave.save(save_file)
print(save_name)
"""
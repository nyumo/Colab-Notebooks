#!/usr/bin/env python
# coding: utf-8
#作成日: 2022-04-15
#改良日: 2022-04-19

import imp
from PIL import Image
import numpy as np
import os #システム操作系
from pathlib import Path #ファイル操作系
import datetime


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
data_path = f'0001_Image2022May16-150116.img'
run_i = 'test'
save_name = f'{run_i}'
save_file = os.path.join(save_folder, save_name + '.png') # 保存先のファイルパス作成

img = Image.open(data_path)
img = img.convert('L')
img.save(save_file)
print(save_name)
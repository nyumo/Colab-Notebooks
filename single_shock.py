# 作成日 2022-08-25

import numpy as np
import os #システム操作系
from pathlib import Path #ファイル操作系
import datetime
#file形式
import csv
import pandas as pd
#グラフ描画
import matplotlib.pyplot as plt

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


#密度ρ0、ユゴニオ[定数項、第一項の係数]
sus304 = [7.93, 4.85, 1.49] #Marsh, 1980
tio2 = [4.23, 3.63, 1.72]   #Syono, 1987
copper = [8.92, 3.91, 1.51] #LASL

P = 5.0 #実験する圧力

def func(a,b,c,x):
    return a*x**2 + b*x + c

def func_2(a,b,c): #二次関数の解の公式 a*x^2 + b*x + c = 0
    x0 = -b - np.sqrt(b**2 - 4*a*c)
    x0 /= 2*a
    x1 = -b + np.sqrt(b**2 - 4*a*c)    
    x1 /= 2*a

    return x0, x1

def func_pq(p,q,a,b,c): #二次関数を(p,q)平行移動
    A = a
    B = -2*p*a + b
    C = a*p**2 - b*p + c + q
    return A, B, C 

def func_rev(u,a,b,c): #x = uで対称移動
    A, B, C = func_pq(2*u, 0, a, -b, c)
    return A, B, C

def func_cross(a0,b0,c0,a1,b1,c1):
    A = a0 - a1
    B = b0 - b1
    C = c0 - c1
    if abs(func_2(A, B, C)[0]) < abs(func_2(A, B, C)[1]):
        x = func_2(A, B, C)[0]
    else:
        x = func_2(A, B, C)[1]

    y = func(a0, b0, c0, x)
    return x, y

def hugoniot(us, a, b):#ユゴニオの式
    up = a + b*us
    return up

def hugoniot_P(rho, a, b):
    A = rho*b
    B = rho*a
    C = 0
    return A, B, C

#試料容器の指定
print('試料容器の素材S or C: ')
capsule = input()
if (capsule == 'S'):
    sus_abc = hugoniot_P(*sus304)
    material = 'SUS304'
else:
    sus_abc = hugoniot_P(*copper)
    material = 'Copper'
tio2_abc = hugoniot_P(*tio2)

#ufsの計算
u = func_2(sus_abc[0], sus_abc[1], sus_abc[2] - P)[1]
print('P, u = '+ str(P) + ',' + str(u))
ufs = 2*u
print('衝突速度ufs = ' + str(ufs))
rev_abc = func_rev(u,*sus_abc)#反転susの係数

#u0,p0の計算
u0 ,p0 = func_cross(*rev_abc, *tio2_abc)
rho = tio2[0]
print('rho, u0, p0 = '+ str(rho) +','+ str(u0)+','+ str(p0))

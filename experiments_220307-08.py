#作成日: 2022-03-22

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

#密度ρ0、ユゴニオ[定数項、第一項の係数]
sus304 = [7.93, 4.85, 1.49] #Marsh, 1980
tio2 = [4.26, 3.63, 1.72]   #Syono, 1987

ufs = 1.37   #衝突速度

P = 30 #実験する圧力
sus_abc = hugoniot_P(*sus304)
tio2_abc = hugoniot_P(*tio2)

#Pの計算
print('衝突速度ufs = ' + str(ufs))
rev_abc = func_rev(ufs*0.5, *sus_abc)#反転susの係数

u = ufs*0.5
P = func(*sus_abc, ufs*0.5)
print('u, P = '+  str(u)+','+ str(P))

#u0,p0の計算
u0 ,p0 = func_cross(*rev_abc, *tio2_abc)
rho = tio2[0]
print('rho, u0, p0 = '+ str(rho) +','+ str(u0)+','+ str(p0))

rho_p = 0.75     #粉末試料密度
tio2_pow = [rho_p, 4.85, 1.49]
ufs = 1.38

tio2_abcp = hugoniot_P(*tio2_pow)

#Pの計算
print('衝突速度ufs = ' + str(ufs))
u = ufs*0.5
P = func(*sus_abc, ufs*0.5)
print('u, P = '+  str(u)+','+ str(P))

#u0,p0の計算
u0p ,p0p = func_cross(*rev_abc, *tio2_abcp)
print('rho, u0, p0 = '+ str(rho_p) +','+ str(u0p)+','+ str(p0p))

#グラフデータ
up = np.arange(0.0, 2.0, 0.01)
tio2_P = func(*tio2_abc, up)
tio2_Pp = func(*tio2_abcp, up)
sus_P = func(*sus_abc, up)
rev_P = func(*rev_abc, up)

data_frame = pd.DataFrame({'rho': np.zeros(up.size),
                    'rhop': np.zeros(up.size),
                    'un': np.zeros(up.size),
                    'pn': np.zeros(up.size),
                    'Up': up,
                    'SUS304_0': sus_P,
                    'SUS304_1': rev_P,
                    'TiO2_0': tio2_P,
                    'TiO2_p0':tio2_Pp})

n = 0
data_frame['rho'][n] = rho
data_frame['rhop'][n] = rho_p
data_frame['un'][n] = u0
data_frame['pn'][n] = p0

dP = P - p0
while dP > 0.1:
    n = n+1
    #u1,p1の計算
    us = hugoniot(u0,tio2[1], tio2[2])
    rho = rho*us
    rho /= us - u0  #rhoの更新

    tio2_abc1 = hugoniot_P(rho, tio2[1], tio2[2])
    tio2_abc1 = func_pq(u0, p0, *tio2_abc1) #平行移動

    if n%2 == 1:
        tio2_abc1 = func_rev(u0, *tio2_abc1)      #対称移動
        u1 ,p1 = func_cross(*sus_abc, *tio2_abc1) #u1,p1の計算
    else:
        u1 ,p1 = func_cross(*rev_abc, *tio2_abc1) #u1,p1の計算
    
    #print(f'rho{n} ,u{n}, p{n} = '+ str(rho) +','+ str(u1)+','+ str(p1))
    tio2_P1 = func(*tio2_abc1, up)
    data_frame[f'TiO2_{n}'] = tio2_P1
    u0 = u1
    p0 = p1

    data_frame['rho'][n] = rho
    data_frame['un'][n] = u0
    data_frame['pn'][n] = p0

    dP = P - p0

print(data_frame)
#csv保存
save_name = dt_now.strftime("%Y-%m-%d_%H-%M-%S_") + 'IMtiO2-sus304'
save_file = os.path.join(save_folder, save_name + '.csv') # 保存先のファイルパス作成
data_frame.to_csv(save_file)
print(save_file)

#pickel保存
save_file = os.path.join(save_folder, save_name + '.pkl') # 保存先のファイルパス作成
data_frame.to_pickle(save_file)
print(save_file)

###############グラフを描画#####################################
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111) 
data_frame.plot(x = 'Up', y = 'SUS304_0', color = 'black', ax = ax)             #プロット 
data_frame.plot(x = 'Up', y = 'SUS304_1', color = 'black', ax = ax)            #プロット 
data_frame.plot(x = 'Up', y = 'TiO2_0', color = 'red', ax = ax)           #プロット
data_frame.plot(x = 'Up', y = 'TiO2_1', color = 'blue', ax = ax)           #プロット
plt.show()
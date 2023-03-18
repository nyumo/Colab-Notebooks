#!/usr/bin/env python
# coding: utf-8
#作成日: 2022-01-26
#改良: 2022-01-27

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
tio2 = [4.26, 3.63, 1.72]   #Syono, 1987

P = 30 #実験する圧力

def func_2(a,b,c): #二次関数の解の公式 a*x^2 + b*x + c = 0
    x0 = -b - np.sqrt(b**2 - 4*a*c)
    x0 /= 2*a
    x1 = -b + np.sqrt(b**2 - 4*a*c)    
    x1 /= 2*a

    return x0, x1

def hugoniot(us, a, b):#ユゴニオの式
    up = a + b*us
    return up

def func_p(rho, a, b, up):
    p = rho*a*up + rho*b*up**2
    return p

def func_rev(P,rho,a,b):
    up = func_2(b, a, -P/rho)[1]
    return up


up = np.arange(0.0, 2.0, 0.1)
sus304_us = hugoniot(up, sus304[1],sus304[2]) 
tio2_us = hugoniot(up, tio2[1], tio2[2])

sus304_p = func_p(*sus304, up)
tio2_p = func_p(*tio2, up)

#ufsの計算
u = func_rev(P, *sus304) #susと反susの交点のy座標: up0, 反susのy切片: ps
print('P, u = '+ str(P) + ',' + str(u))
ufs = 2*u
print('衝突速度ufs = ' + str(ufs))

sus_rev = func_p(*sus304, -up + ufs)

#(u0,po)の計算
def func_u0p0(u,rhoa,a0,a1,rhob,b0,b1):
    a = rhoa*a1 - rhob*b1
    b = -2*rhoa*a1*2*u - rhoa*a0 - rhob*b0
    c = rhoa*a1*((2*u)**2) + rhoa*a0*2*u
    
    u0 = func_2(a, b, c)[0]
    p0 = func_p(rhob, b0, b1, u0)
    return u0,p0

def func_u1p1(u,p,rhoa,a0,a1,rhob,b0,b1):
    a = rhoa*a1 - rhob*b1
    b = -2*rhoa*a1*u - rhoa*a0 - rhob*b0
    c = rhoa*a1*(u**2) + rhoa*a0*u + p
    
    u0 = func_2(a, b, c)[0]
    p0 = func_p(rhob, b0, b1, u0)
    return u0,p0

def func_u2p2(u,p,rhoa,a0,a1,rev0, rev1, rev2):
    a = rhoa*a1 - rev0
    b = -2*rhoa*a1*u + rhoa*a0 - rev1
    c = rhoa*a1*(u**2) - rhoa*a0*u + p - rev2 
    
    u0 = func_2(a, b, c)[0]
    p0 = rev0*u0**2 + rev1*u0 + rev2
    return u0,p0

u0, p0 = func_u0p0(u, *sus304, *tio2)
print('u0, p0 = '+ str(u0)+','+ str(p0))
rev = [sus304[0]*sus304[2],
         -2*sus304[0]*sus304[2]*2*u0 - sus304[0]*sus304[1],
         sus304[0]*sus304[2]*(2*u0)**2 + sus304[0]*sus304[1]*2*u0] #反susの二次関数a, b, c

#u1, P1の計算
n = 0
rho = tio2[0]
dP = P - p0

#while dP > 0.1**2:
n = n+1
us = hugoniot(u0,tio2[1], tio2[2])
#print(us)
rho = rho*us
rho /= us - u0
if n %2 == 1:#sus304の交点
    u1, p1 = func_u1p1(u0,p0, rho, tio2[1], tio2[2], *sus304)
else:#sus304の交点
    u1, p1 = func_u2p2(u0,p0, rho, tio2[1], tio2[2], *rev)

print(f'u{n}, p{n} = '+ str(u1)+','+ str(p1))
tio2_rev = func_p(rho, tio2[1], tio2[2], -up + u0) + p0
u0 = u1
p0 = p1
dP = P - p0

n = n+1
us = hugoniot(u0,tio2[1], tio2[2])
#print(us)
rho = rho*us
rho /= us - u0
if n %2 == 1:#sus304の交点
    u1, p1 = func_u1p1(u0,p0, rho, tio2[1], tio2[2], *sus304)
else:#sus304の交点
    u1, p1 = func_u2p2(u0,p0, rho, tio2[1], tio2[2], *rev)

print(f'u{n}, p{n} = '+ str(u1)+','+ str(p1))
tio2_rev = func_p(rho, tio2[1], tio2[2], -up + u0) + p0


data_frame = pd.DataFrame({'Up': up,
                    'TiO2_0': tio2_p,
                    'SUS304_0': sus304_p,
                    'TiO2_1': tio2_rev,
                    'SUS304_1': sus_rev})

#print(data_frame)

#csv保存
#save_name = dt_now.strftime("%Y-%m-%d_%H-%M-%S_") + 'IMtiO2-sus304'
#save_file = os.path.join(save_folder, save_name + '.csv') # 保存先のファイルパス作成
#data_frame.to_csv(save_file)
#print(save_file)

#pickel保存
#save_file = os.path.join(save_folder, save_name + '.pkl') # 保存先のファイルパス作成
#data_frame.to_pickle(save_file)
#print(save_file)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
data_frame.plot(x = 'Up', y = 'TiO2_0', color = 'red', ax = ax)           #プロット 
data_frame.plot(x = 'Up', y = 'SUS304_0', color = 'black', ax = ax)             #プロット 
data_frame.plot(x = 'Up', y = 'SUS304_1', color = 'black', ax = ax)            #プロット 
data_frame.plot(x = 'Up', y = 'TiO2_1', color = 'red', ax = ax)           #プロット 
plt.show()

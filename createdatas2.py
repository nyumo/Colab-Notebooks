import numpy as np
from scipy.optimize import curve_fit    # フィッティング用
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import os
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
import time
import csv
from mpl_toolkits.mplot3d import Axes3D

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #意味はわからん

# ガウシアンビームの関数の定義
def gaussian_beam(x,a,b,c,d):
    return  a * np.exp(-2*(x-b)*(x-b)/c/c) + d

def createdata1(N, NOISE, data_size, x, y, tx, ty, fit_param_x, fit_param_y):
    start = time.time()
    
    # ガウシアンビームのパラメータ
    i0 = 1.0
    x0 = 0.0
    y0 = 0.0
    w0 = 5.0
    h0 = 0.0
    param_ini_x = np.array([i0, x0, w0, h0])  # フィッティングの初期値 (ここではデータから推定は行わない)
    param_ini_y = np.array([i0, y0, w0, h0])
    t = np.arange(0.0, 2.0, 0.1).repeat(N/20)
    k = 1

    for n in range(N):        # 繰り返し精度を調べるために各ノイズ割合でN回実行
        
        center = data_size/2
        # x配列とy配列
        x_array = np.arange(-center, center, 1.0)                         # x配列
        y_array = np.arange(-center, center, 1.0)                         # y配列
        nx = len(x_array)
        ny = len(y_array)
        intensity = np.zeros((nx, ny))                            # ノイズを含まない2次元強度分布
        x0 = t[n] -1
        x_grid, y_grid = np.meshgrid(x_array, y_array)
        intensity = i0 * np.exp(-2*((x_grid-x0)**2 + (y_grid-y0)**2)/w0**2).T
        
        # 最大強度を取る位置における強度プロファイル
        profile_x = np.zeros(nx)
        profile_y = np.zeros(ny)
    

        # 2次元の強度分布にノイズを付与
        noise = (np.random.rand(nx*ny)-0.5)*i0*NOISE*0.01   #プラスマイナスNOISE%のノイズ(一様分布), (np.random.rand(nx*ny)-0.5)*2の部分が-1から1までの乱数になる
        noise = noise.reshape((nx,ny))
        intensity_noise = intensity + noise

        # 最大値の探索 & その位置の強度プロファイルの取得
        idx = np.unravel_index(np.argmax(intensity_noise), intensity_noise.shape)
        profile_x = intensity_noise[:,idx[1]]
        profile_y = intensity_noise[idx[0],:]
        x[n] = profile_x
        y[n] = profile_y
        tx[n] = (i0, t[n], w0)
        ty[n] = (i0, y0, w0)

        # x方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, x_array, profile_x, p0=param_ini_x, maxfev=2000)
        fit_param_x[n][0] = param[0]
        fit_param_x[n][1] = param[1]
        fit_param_x[n][2] = param[2]
        fit_param_x[n][3] = param[3]
        
        # y方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, y_array, profile_y, p0=param_ini_y, maxfev=2000)
        fit_param_y[n][0] = param[0]
        fit_param_y[n][1] = param[1]
        fit_param_y[n][2] = param[2]
        fit_param_y[n][3] = param[3]
        
        if n == N-1:
            elapsed_time = time.time() - start
            print ("経過時間:{0}".format(elapsed_time) + "[sec]")
            fit_ave = np.average(abs(fit_param_x.T[1]-tx.T[1]+1))
            fit_std = np.std(abs(fit_param_x.T[1]-tx.T[1]+1))
            fitting = "平均絶対誤差: %f ± %f" % (fit_ave, fit_std)
            print(fitting)
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("x", fontsize=16)
            ax.set_ylabel("y", fontsize=16)
            ax.set_zlabel("Intensity [a.u.]", fontsize=16)
            plt.tick_params(labelsize=16)
            ax.grid(False)
            ax.plot_wireframe(x_grid, y_grid, intensity_noise, color='black', linewidth=0.3)
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            plt.show()

        
    return N, x, y, tx, ty, fit_param_x, fit_param_y

#中心座標動かない方
def createdata2(N, NOISE, data_size, x, y, tx, ty, fit_param_x, fit_param_y):
    start = time.time()
    
    # ガウシアンビームのパラメータ
    i0 = 1.0
    x0 = 0.0
    y0 = 0.0
    w0 = 5.0
    h0 = 0.0
    param_ini_x = np.array([i0, x0, w0, h0])  # フィッティングの初期値 (ここではデータから推定は行わない)
    param_ini_y = np.array([i0, y0, w0, h0])
    
    k = 1

    for n in range(N):        # 繰り返し精度を調べるために各ノイズ割合でN回実行
        
        center = data_size/2
        # x配列とy配列
        x_array = np.arange(-center, center, 1.0)                         # x配列
        y_array = np.arange(-center, center, 1.0)                         # y配列
        nx = len(x_array)
        ny = len(y_array)
        intensity = np.zeros((nx, ny))                            # ノイズを含まない2次元強度分布
        x_grid, y_grid = np.meshgrid(x_array, y_array)
        intensity = i0 * np.exp(-2*((x_grid-x0)**2 + (y_grid-y0)**2)/w0**2).T
        
        # 最大強度を取る位置における強度プロファイル
        profile_x = np.zeros(nx)
        profile_y = np.zeros(ny)
    

        # 2次元の強度分布にノイズを付与
        noise = (np.random.rand(nx*ny)-0.5)*i0*NOISE*0.01   #プラスマイナスNOISE%のノイズ(一様分布), (np.random.rand(nx*ny)-0.5)*2の部分が-1から1までの乱数になる
        noise = noise.reshape((nx,ny))
        intensity_noise = intensity + noise

        # 最大値の探索 & その位置の強度プロファイルの取得
        idx = np.unravel_index(np.argmax(intensity_noise), intensity_noise.shape)
        profile_x= intensity_noise[:,idx[1]]
        profile_y = intensity_noise[idx[0],:]
        x[n] = profile_x
        y[n] = profile_y
        tx[n] = (i0, x0+1.0, w0)
        ty[n] = (i0, y0, w0)
        # x方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, x_array, profile_x, p0=param_ini_x, maxfev=2000)
        fit_param_x[n][0] = param[0]
        fit_param_x[n][1] = param[1]
        fit_param_x[n][2] = param[2]
        fit_param_x[n][3] = param[3]
        
        # y方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, y_array, profile_y, p0=param_ini_y, maxfev=2000)
        fit_param_y[n][0] = param[0]
        fit_param_y[n][1] = param[1]
        fit_param_y[n][2] = param[2]
        fit_param_y[n][3] = param[3]
        
        if n == N-1:
            elapsed_time = time.time() - start
            print ("経過時間:{0}".format(elapsed_time) + "[sec]")
            fit_ave = np.average(abs(fit_param_x.T[1]-tx.T[1]+1))
            fit_std = np.std(abs(fit_param_x.T[1]-tx.T[1]+1))
            fitting = "平均絶対誤差: %f ± %f" % (fit_ave, fit_std)
            print(fitting)
    return N, x, y, tx, ty, fit_param_x, fit_param_y

def createdata3(N, NOISE, data_size, x, y, tx, ty, fit_param_x, fit_param_y):
    start = time.time()
    
    # ガウシアンビームのパラメータ
    i0 = 1.0
    x0 = 0.0
    y0 = 0.0
    w0 = 5.0
    h0 = 0.0
    param_ini_x = np.array([i0, x0, w0, h0])  # フィッティングの初期値 (ここではデータから推定は行わない)
    param_ini_y = np.array([i0, y0, w0, h0])
    t = np.arange(0.5, 1.5, 0.1).repeat(N/10)
    k = 1

    for n in range(N):        # 繰り返し精度を調べるために各ノイズ割合でN回実行
        
        center = data_size/2
        # x配列とy配列
        x_array = np.arange(-center, center, 1.0)                         # x配列
        y_array = np.arange(-center, center, 1.0)                         # y配列
        nx = len(x_array)
        ny = len(y_array)
        intensity = np.zeros((nx, ny))                            # ノイズを含まない2次元強度分布
        x0 = t[n] -1
        x_grid, y_grid = np.meshgrid(x_array, y_array)
        intensity = i0 * np.exp(-2*((x_grid-x0)**2 + (y_grid-y0)**2)/w0**2).T
        
        # 最大強度を取る位置における強度プロファイル
        profile_x = np.zeros(nx)
        profile_y = np.zeros(ny)
    

        # 2次元の強度分布にノイズを付与
        noise = (np.random.rand(nx*ny)-0.5)*i0*NOISE*0.01   #プラスマイナスNOISE%のノイズ(一様分布), (np.random.rand(nx*ny)-0.5)*2の部分が-1から1までの乱数になる
        noise = noise.reshape((nx,ny))
        intensity_noise = intensity + noise

        # 最大値の探索 & その位置の強度プロファイルの取得
        idx = np.unravel_index(np.argmax(intensity_noise), intensity_noise.shape)
        profile_x = intensity_noise[:,idx[1]]
        profile_y = intensity_noise[idx[0],:]
        x[n] = profile_x
        y[n] = profile_y
        tx[n] = (i0, t[n], w0)
        ty[n] = (i0, y0, w0)

        # x方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, x_array, profile_x, p0=param_ini_x, maxfev=2000)
        fit_param_x[n][0] = param[0]
        fit_param_x[n][1] = param[1]
        fit_param_x[n][2] = param[2]
        fit_param_x[n][3] = param[3]
        
        # y方向のプロファイルの非線形フィッティング
        param, cov  = curve_fit(gaussian_beam, y_array, profile_y, p0=param_ini_y, maxfev=2000)
        fit_param_y[n][0] = param[0]
        fit_param_y[n][1] = param[1]
        fit_param_y[n][2] = param[2]
        fit_param_y[n][3] = param[3]
        
        if n == N-1:
            elapsed_time = time.time() - start
            print ("経過時間:{0}".format(elapsed_time) + "[sec]")
            fit_ave = np.average(abs(fit_param_x.T[1]-tx.T[1]+1))
            fit_std = np.std(abs(fit_param_x.T[1]-tx.T[1]+1))
            fitting = "平均絶対誤差: %f ± %f" % (fit_ave, fit_std)
            print(fitting)
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("x", fontsize=16)
            ax.set_ylabel("y", fontsize=16)
            ax.set_zlabel("Intensity [a.u.]", fontsize=16)
            plt.tick_params(labelsize=16)
            ax.grid(False)
            ax.plot_wireframe(x_grid, y_grid, intensity_noise, color='black', linewidth=0.3)
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            plt.show()

        
    return N, x, y, tx, ty, fit_param_x, fit_param_y


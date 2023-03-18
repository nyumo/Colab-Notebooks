# ----------------------------------------------------
# ガウスフィッティング
# ノイズを付加したときのガウスフィッティングの精度を求める
# ----------------------------------------------------

# 必要なパッケージをインポート
from scipy.optimize import curve_fit    # フィッティング用
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #意味はわからん

#学習パラメータの取得
save_model_path = "/Users/nagaiyuma/Desktop/maindata/model.h5"
model = load_model(save_model_path)

# ガウシアンビームの関数の定義
def gaussian_beam(x,a,b,c,d):
    return  a * np.exp(-2*(x-b)*(x-b)/c/c) + d

# ガウシアンビームのパラメータ
i0 = 1.0
x0 = 0.0
y0 = 0.0
w0 = 2.5
h0 = 0.0
param_ini_x = np.array([i0, x0, w0, h0])  # フィッティングの初期値 (ここではデータから推定は行わない)
param_ini_y = np.array([i0, y0, w0, h0])

# x配列とy配列
x_array = np.arange(-50, 50, 1.0)                         # x配列
y_array = np.arange(-50, 50, 1.0)                         # y配列
nx = len(x_array)
ny = len(y_array)
intensity = np.zeros((nx, ny))                            # ノイズを含まない2次元強度分布
for i in range(nx):
    for j in range(ny):
        intensity[i][j] = i0 * np.exp(-2*((x_array[i]-x0)*(x_array[i]-x0) + (y_array[j]-y0)*(y_array[j]-y0))/w0/w0)

# 最大強度を取る位置における強度プロファイル
profile_x = np.zeros(nx)
profile_y = np.zeros(ny)

# 信号に対するノイズ量[%]
NOISE = 30

# N回繰り返しフィッティングを行う
N = 1000
n_N      = np.zeros(N)
x0_N     = np.zeros(N)   # N回分のx0,w0を格納する配列
y0_N     = np.zeros(N)
w0_x_N   = np.zeros(N)
w0_y_N   = np.zeros(N)
h0_x_N   = np.zeros(N)
h0_y_N   = np.zeros(N)
predx0_N = np.zeros(N)
predy0_N = np.zeros(N)

for n in range(N):        # 繰り返し精度を調べるために各ノイズ割合でN回実行
    
    n_N[n] = n + 1
    
    # 2次元の強度分布にノイズを付与
    noise = (np.random.rand(nx*ny)-0.5)*i0*NOISE*0.01   # プラスマイナスNOISE%のノイズ(一様分布), (np.random.rand(nx*ny)-0.5)*2の部分が-1から1までの乱数になる
    noise = noise.reshape((nx,ny))
    intensity_noise = intensity + noise
    
    # 最大値の探索 & その位置の強度プロファイルの取得
    idx = np.unravel_index(np.argmax(intensity_noise), intensity_noise.shape)
    profile_x = intensity_noise[:,idx[1]]
    profile_y = intensity_noise[idx[0],:]
    
    # x方向のプロファイルの非線形フィッティング
    param, cov  = curve_fit(gaussian_beam, x_array, profile_x, p0=param_ini_x, maxfev=2000)
    i0_x        = param[0]
    x0_N[n]     = param[1]
    w0_x_N[n]   = param[2]
    h0_x_N[n]   = param[3]
    # y方向のプロファイルの非線形フィッティング
    param, cov  = curve_fit(gaussian_beam, y_array, profile_y, p0=param_ini_y, maxfev=2000)
    i0_y        = param[0]
    y0_N[n]     = param[1]
    w0_y_N[n]   = param[2]
    h0_y_N[n]   = param[3]
    #x方向のプロファイルのCNN
    cnn_x       = profile_x.reshape(-1,100,1)
    predx0_N[n] = model.predict(cnn_x).flatten()[0] -1
    #y方向のプロファイルのCNN
    cnn_y       = profile_y.reshape(-1,100,1)
    predy0_N[n] = model.predict(cnn_y).flatten()[0] -1
    

# 最後の結果だけグラフにする
profile_x_fit = gaussian_beam(x_array, i0_x, x0_N[N-1], w0_x_N[N-1], h0_x_N[N-1])  # 最尤フィッティング配列の計算
profile_y_fit = gaussian_beam(y_array, i0_y, y0_N[N-1], w0_y_N[N-1], h0_y_N[N-1])
savename = "noise%d_N%d.png" % (NOISE, N)
plt.figure()
plt.subplots_adjust(hspace=0.5)
# x方向の強度プロファイル
plt.subplot(2,1,1)
plt.scatter(x_array, profile_x, color="black", label="x-profile")  # ノイズありのプロット
plt.plot(x_array, profile_x_fit, color="red", label="x-fitting")   # フィッティングカーブの描画
plt.legend()             # 凡例
plt.xlabel('x')          # 軸ラベル
plt.ylabel('Intensity')
plt.ylim(-1.0, 2.0)      # y軸の表示範囲
# y方向の強度プロファイル
plt.subplot(2,1,2)
plt.scatter(y_array, profile_y, color="black", label="y-profile")
plt.plot(y_array, profile_y_fit, color="blue", label="y-fitting")
plt.legend()
plt.xlabel('y')
plt.ylabel('Intensity')
plt.ylim(-1.0, 2.0)
plt.savefig(savename) # png画像として出力
plt.close()

# ヒストグラムのグラフを作成
plt.figure()
savename = "f_noise%d_N%d_x0_histogram.png" % (NOISE, N)
plt.hist(x0_N, bins=30, color='black')
plt.xlabel('x0')
plt.ylabel('Frequency')
plt.xlim(-1.0, 1.0)
plt.savefig(savename)
plt.close()

# ヒストグラムのグラフを作成
plt.figure()
savename = "c_noise%d_N%d_x0_histogram.png" % (NOISE, N)
plt.hist(predx0_N, bins=30, color='black')
plt.xlabel('x0')
plt.ylabel('Frequency')
plt.xlim(-1.0, 1.0)
plt.savefig(savename)
plt.close()

title = [["n","x0","y0","w0_x","w0_y"]]                # 凡例のラベル
all = np.arange(N*5, dtype='float64').reshape((N,5))   # 1次元配列をまとめるための2次元配列
all[0:, 0] = n_N                                       # 2次元配列の該当の列のみ書き換え (この方法がいいかは知らん)
all[0:, 1] = x0_N
all[0:, 2] = y0_N
all[0:, 3] = w0_x_N
all[0:, 4] = w0_y_N


# ファイル出力 (追記モード)
savename = "noise%d_N%d_alldata.txt" % (NOISE, N)
with open(savename, 'a') as fp:
    np.savetxt(fp, title, delimiter='\t', fmt='%s')
    np.savetxt(fp, all, delimiter='\t', fmt='%.6f')



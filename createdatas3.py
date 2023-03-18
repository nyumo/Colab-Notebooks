import numpy as np
import os

# ガウシアンビームの関数の定義
def gaussian_beam(x,a,b,c,d):
    return  a * np.exp(-2*(x-b)*(x-b)/c/c) + d

def intensity(x_array, y_array, param):
    i0 = param[0]
    x0 = param[1]
    y0 = param[2]
    w0 = param[3]
    h0 = param[4]
    # x配列とy配列
    nx = len(x_array)
    ny = len(y_array)
    intensity = np.zeros((nx, ny))
    x_grid, y_grid = np.meshgrid(x_array, y_array)
    intensity = i0 * np.exp(-2*((x_grid-x0)**2 + (y_grid-y0)**2)/w0**2).T
    return intensity
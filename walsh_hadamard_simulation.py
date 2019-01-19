# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 01:11:35 2019

@author: ohman
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def img_write(input_img, name):
    input_img = np.where(input_img==1, 255, 0)
    input_img = input_img.astype("uint8")
    cv2.imwrite("img/" + name + ".bmp",input_img)


def img_show(input_img):
    input_img = np.where(input_img==1, 255, 0)
    input_img = input_img.astype("uint8")
    cv2.imshow("img", input_img)
    cv2.waitKey(0)

#アダマールマスク作成
def make_hadamard(shape):
    
    # 2のn乗かどうか確認
    n = np.log(shape) / np.log(2)
    if 2**n == shape:
        print("making hadamard")
    else:
        print("error")
        return None, None
    
    hadamard_shape = [2**i for i in range(int(n) +1)]
    print("hadamard shape\n{}".format(hadamard_shape))

    hadamard = np.array([1])
    hadamard_matrix = {1 : hadamard}
    iteration = len(hadamard_shape) -1
    for i in range(iteration):
        hadamard = np.hstack((hadamard,hadamard))
        hadamard = np.vstack((hadamard,hadamard))
        #反転させるインデックス (4,4) なら最後の(2,2)を反転させる
        reverse_range = -hadamard_shape[i]
        hadamard[reverse_range:,reverse_range:] = hadamard[reverse_range:,reverse_range:] * -1
        hadamard_matrix[hadamard_shape[i+1]] = hadamard
        
    return hadamard_matrix

#各行でそれぞれ，符号が変わった回数を数える
def change_count(shape, hadamard_matrix):
    encode = hadamard_matrix[shape]
    zero_crossing = np.zeros(shape)
    for h in range(shape):
        for w in range(shape-1):
            if encode[h][w] != encode[h][w+1]:
                zero_crossing[h] += 1
    return zero_crossing

#ウォルシュアダマールに変換(符号が変わる回数が少ない行順に並べ替え)
def make_walsh_hadamard(shape, hadamard_matrix, zero_crossing):
    walsh_hadamard = hadamard_matrix[shape].copy()
    encode = hadamard_matrix[shape]
    
    #zero_crossing が小さい順でインデックスを取得
    indexes = np.argsort(zero_crossing)

    for i,index in enumerate(indexes):
        walsh_hadamard[i] = encode[index]
    
    return walsh_hadamard


def RMSE(img1,img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif ** 2
    rmse = np.sqrt(np.sum(dif2) / (n))
    return rmse

def simulation(img_array, walsh_hadamard, height,width):
    errors = []
    #encode = walsh_hadamard
    re_imges = {}
    for i in range(height*width):
        tank = walsh_hadamard[0:i+1,:]
        print("tank",tank.shape)
        mask_inv = np.linalg.pinv(tank)
        print("mask_inv",mask_inv.shape)
        output_array = np.dot(tank , img_array)
        print("output_array",output_array.shape)
        reconstruct = np.dot(mask_inv,output_array)
        print("reconstruct",reconstruct.shape)
        reimg = reconstruct.reshape(height,width).astype("uint8")
        re_imges[i+1] = reimg
        error = RMSE(img_array, reconstruct)
        errors.append(error)
        print(i)
    return errors, re_imges


def error_plot(errors):
    plt.ylabel("RMSE" , fontsize=15)
    plt.xlabel("iteration number", fontsize=15)
    plt.ylim(0,max(errors)+10)
    plt.plot(np.arange(1,len(errors)+1),errors)


def main():
    
    img = cv2.imread("img/sample_256pixel.bmp",0)
    print(img.shape)
    height, width = img.shape[:2]
    print("height : {}\nwidth : {}".format(height,width))
    
    #img_array = img.reshape(height*width ,1)
    shape = height * width
    img_array = img.reshape(height*width, 1)
    print(img_array.shape)
    
    
    #アダマールマスク作成
    hadamard_matrix = make_hadamard(shape)
    
    #各行でそれぞれ，符号が変わった回数を数える
    zero_crossing = change_count(shape, hadamard_matrix)
    print(len(zero_crossing))
    #ウォルシュアダマールに変換(符号が変わる回数が少ない行順に並べ替え)
    walsh_hadamard = make_walsh_hadamard(shape, hadamard_matrix, zero_crossing)
    
    
    
    img_show(walsh_hadamard)
    
    name = "walsh_hadamard" + str(shape)
    img_write(walsh_hadamard, name)


    errors, re_imges = simulation(img_array, walsh_hadamard, height, width)
    error_plot(errors)


if __name__ == "__main__":
    main()
    

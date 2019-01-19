# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:12:36 2019

@author: ohman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#符号化行列の生成
def make_encode(height,width):
    while True:
        #0~2の範囲でランダム行列(h^2,w^2)  (36,36)
        encoded = 2 * np.random.rand(height**2,width**2)

        #画素値が閾値より大きければある値(白)を割り当て，そうでなければ別の値(黒)を割り当てる
        ret, encoded = cv2.threshold(encoded, 0.5 , 1 , cv2.THRESH_BINARY)
        
        if cv2.determinant(encoded) != 0:
            break
    return encoded

#root mean square error
def RMSE(img1,img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif ** 2
    rmse = np.sqrt(np.sum(dif2) / (n))
    return rmse

def main():
    #img (6,6)
    img = cv2.imread("img/sample_36pixel.bmp",0)
    #print("img shape" , img.shape)
    height, width = img.shape[:2]
    #print("height : {}\nwidth : {}".format(height,width))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    #img_array (h,w)→(h*w,1) (36,1)
    img_array = img.reshape(height*width ,1)
    #print("img_array shape" , img_array.shape)
    cv2.imshow("img_array", img_array)
    cv2.waitKey(0)
    
    #符号化行列 (h,w) (36,36)
    encoded = make_encode(height,width)
    #print(encoded.shape)
    cv2.imshow("encoded", encoded)
    cv2.waitKey(0)
        
    errors = []
    for i in range(height*width):
        tank = encoded[0:i+1,:]
        #print("tank",tank.shape) (1,36)→(2,36),・・・(36,36)
        
        mask_inv = np.linalg.pinv(tank)
        #print("mask_inv",mask_inv.shape) (36,1)→(36,2)・・・(36,36)
        
        output_array = np.dot(tank , img_array)
        #print("output_array",output_array.shape) (i+1,36)・(36,1) = (i+1,1) 
        
        reconstruct = np.dot(mask_inv,output_array)
        #print("reconstruct",reconstruct.shape) (36,i+1)・(i+1,1) = (36,1)
        
        reimg = reconstruct.reshape(height,width).astype("uint8")
        #print("reimg",reimg.shape) (6,6)
        
        #cv2.imwrite("result/reconstruct_{}.bmp".format(i+1) , reimg)
        
        error = RMSE(img_array, reconstruct)
        errors.append(error)
    
    cv2.imshow("reconstruct img", reimg)
    cv2.waitKey(0)
    
    plt.figure(figsize=(12, 8)) 
    plt.ylabel("RMSE" , fontsize=25)
    plt.xlabel("iteration number", fontsize=25)
    plt.ylim(0,max(errors)+10)
    plt.tick_params(labelsize=20)
    plt.grid(which='major',color='black',linestyle='-')
    plt.plot(np.arange(1,37),errors)
    plt.show()


if __name__ == "__main__":
    main()

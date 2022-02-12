'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 17:52:15
LastEditors: Zhang
LastEditTime: 2021-12-04 16:43:10
'''

import numpy as np
from PIL import ImageGrab
import cv2
import time
import grabscreen
import os

def self_blood_count(self_gray):
    self_blood = 0
    for self_bd_num in self_gray[533]:
        # self blood gray pixel 80~98
        # 血量灰度值80~98
        #print(self_bd_num)
        if self_bd_num > 90 and self_bd_num < 98:
            self_blood += 1
    return self_blood

def boss_blood_count(boss_gray):
    boss_blood = 0
    for boss_bd_num in boss_gray[0]:
    # boss blood gray pixel 65~75
    # 血量灰度值65~75 
        #print(boss_bd_num)
        if boss_bd_num > 65 and boss_bd_num < 80:
            boss_blood += 1
    return boss_blood

def self_stamina_count(self_gray):
    self_stamina = 0
    for self_bd_num in self_gray[519]:
        # self blood gray pixel 80~98
        #血量灰度值80~98
        #print(self_bd_num)
        if self_bd_num > 134 and self_bd_num < 137:
            self_stamina += 1
        elif self_bd_num > 175 and self_bd_num < 186:
            self_stamina += 1
    return self_stamina

def boss_stamina_count(boss_gray):
    boss_stamina = 0
    for boss_bd_num in boss_gray[0]:
    # boss blood gray pixel 65~75
    # 血量灰度值65~75 
        #print(boss_bd_num)
        if boss_bd_num > 134 and boss_bd_num < 137:
            boss_stamina += 1
        elif boss_bd_num > 170 and boss_bd_num < 178:
            boss_stamina += 1
    return boss_stamina

wait_time = 5
L_t = 3

obs_window = (335,90,835,590)#384,344  192,172 96,86
#blood_window = (75,95,455,630)
blood_window = (75,97,255,630)

stamina_window = (586,79,792,598)

for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):

    #printscreen = np.array(ImageGrab.grab(bbox=(window_size)))
    #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    #pil格式耗时太长
    
    blood_gray = cv2.cvtColor(grabscreen.grab_screen(blood_window),cv2.COLOR_BGR2GRAY)#灰度图像收集
    
    obs_screen = grabscreen.grab_screen(obs_window)#状态彩色图像
    
    obs_resize = cv2.resize(obs_screen,(150,150))
    obs_bgr = cv2.cvtColor(obs_resize,cv2.COLOR_BGR2RGB)
    obs_rgb = cv2.cvtColor(obs_bgr,cv2.COLOR_BGR2RGB)
    obs = np.array(obs_rgb).reshape(-1,150,150,3)[0]


    self_blood = self_blood_count(blood_gray)
    #print("己方血量:",self_blood)
    boss_blood = boss_blood_count(blood_gray)
    #print("Boss血量:",boss_blood)

    stamina_gray = cv2.cvtColor(grabscreen.grab_screen(stamina_window),cv2.COLOR_BGR2GRAY)#灰度图像收集
    # screen_reshape = cv2.resize(screen_gray,(96,86))
    self_stamina = self_stamina_count(stamina_gray)
    #print("己方架势条:",self_stamina)
    boss_stamina = boss_stamina_count(stamina_gray)   
    #print("Boss架势条:",boss_stamina) 
    
    #cv2.imshow('window1',blood_gray)
    #cv2.imshow('window2',stamina_gray)
    cv2.imshow('window3',obs)
    print(obs.shape)
    #cv2.imshow('window2',screen_reshape)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
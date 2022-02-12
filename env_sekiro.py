'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-16 17:15:06
LastEditors: Zhang
LastEditTime: 2021-12-08 22:13:33
'''
import numpy as np
import torch
from observation.grabscreen import grab_screen
import cv2
import time
import utils.directkeys as directkeys
from utils.getkeys import key_check
from utils.restart import restart


class Sekiro(object):
    def __init__(self, observation_w, observation_h, action_dim):
        super().__init__()

        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h

        self.action_dim = action_dim

        self.obs_window = (335,90,835,590)
        self.blood_window = (75,97,255,630)
        self.stamina_window = (586,79,792,598)

        self.boss_blood = 0
        self.self_blood = 0
        self.boss_stamina = 0
        self.self_stamina = 0

        self.stop = 0
        self.emergence_break = 0

    def self_blood_count(self, self_gray):
        self_blood = 0
        for self_bd_num in self_gray[533]:
            # self blood gray pixel 80~98
            # 血量灰度值80~98
            #print(self_bd_num)
            if self_bd_num > 90 and self_bd_num < 98:
                self_blood += 1
        return self_blood

    def boss_blood_count(self, boss_gray):
        boss_blood = 0
        for boss_bd_num in boss_gray[0]:
            # boss blood gray pixel 65~75
            # 血量灰度值65~75 
            #print(boss_bd_num)
            if boss_bd_num > 65 and boss_bd_num < 75:
                boss_blood += 1
        return boss_blood

    def self_stamina_count(self, self_gray):
        self_stamina = 0
        for self_bd_num in self_gray[519]:
            # self blood gray pixel 80~98
            #架势条灰度值80~98
            #print(self_bd_num)
            if self_bd_num > 135 and self_bd_num < 145:
                self_stamina += 1
            elif self_bd_num > 175 and self_bd_num < 186:
                self_stamina += 1
        return self_stamina

    def boss_stamina_count(self, boss_gray):
        boss_stamina = 0
        for boss_bd_num in boss_gray[0]:
        # boss blood gray pixel 65~75
        # 血量灰度值65~75 
        #print(boss_bd_num)
            if boss_bd_num > 135 and boss_bd_num < 145:
                boss_stamina += 1
            elif boss_bd_num > 170 and boss_bd_num < 178:
                boss_stamina += 1
        return boss_stamina
    
    def take_action(self, action):
        if action == 0: #j
            directkeys.attack()
        elif action == 1: #m
            directkeys.defense()
        elif action == 2: #k
            directkeys.jump()
        elif action == 3: #r
            directkeys.dodge_forward()
        #elif action == 4: #r_back
           # directkeys.dodge_back()
       # elif action == 5: #hard_attack
           # directkeys.hard_attack()
        #elif action == 1: #ninja_attack
           # directkeys.ninja_attack()
        #elif action == 3: #skill_attack
            #directkeys.skill_attack()
        
    def get_reward(self, boss_blood, next_boss_blood, self_blood, next_self_blood, 
                   boss_stamina, next_boss_stamina, self_stamina, next_self_stamina, 
                   stop, emergence_break,
                   self_blood_gamma=0.6, boss_blood_gamma=0.4, self_stamina_gamma=0.5, boss_stamina_gamma=0.5):
        if next_self_blood < 3:     # self dead
            if emergence_break < 2:
                reward = -20
                done = 1
                stop = 0
                emergence_break += 1
                return reward, done, stop, emergence_break
            else:
                reward = -20
                done = 1
                stop = 0
                emergence_break = 100
                return reward, done, stop, emergence_break
        elif next_boss_blood - boss_blood > 15  or next_boss_blood < 3:   #boss dead
            if emergence_break < 2:
                reward = 30
                done = 1
                stop = 0
                emergence_break += 1
                return reward, done, stop, emergence_break
            else:
                reward = 30
                done = 1
                stop = 0
                emergence_break = 100
                return reward, done, stop, emergence_break
        
        else:
            self_blood_reward = 0
            boss_blood_reward = 0
            self_stamina_reward = 0
            boss_stamina_reward = 0
            # print(next_self_blood - self_blood)
            # print(next_boss_blood - boss_blood)
            if next_self_blood - self_blood < -5:
                if stop == 0:
                    self_blood_reward = -10
                    stop = 1
                # 防止连续取帧时一直计算掉血
            else:
                stop = 0
            if next_boss_blood - boss_blood <= -2:
                boss_blood_reward = 10
            
            if next_self_stamina - self_stamina >=2:
                self_stamina_reward = -10
            
            if next_boss_stamina - boss_stamina >=2:
                boss_stamina_reward = 10
            # print("self_blood_reward:    ",self_blood_reward)
            # print("boss_blood_reward:    ",boss_blood_reward)
            reward = self_blood_gamma * self_blood_reward + boss_blood_gamma * boss_blood_reward + self_stamina_gamma * self_stamina_reward + boss_stamina_gamma * boss_stamina_reward
            done = 0
            emergence_break = 0
        return reward, done, stop, emergence_break

    def step(self, action):
        # if action == 2:
        #     loss_reward = -0.5
        if action == 2:
            loss_reward = -0.5
        elif action == 0:
            loss_reward = 0.5
        elif action == 3:
            loss_reward = -0.5
        # elif action == 6:
        #     loss_reward = -1
        # elif action == 7:
        #     loss_reward = -0.5
        # #elif action == 2:
        #     #loss_reward = -0.5
        else:
            loss_reward = 0
        self.take_action(action)
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen,(self.width,self.height))
        obs_bgr = cv2.cvtColor(obs_resize,cv2.COLOR_BGR2RGB)
        obs_rgb = cv2.cvtColor(obs_bgr,cv2.COLOR_BGR2RGB)
        #obs_gay = cv2.cvtColor(obs_resize, cv2.COLOR_BGR2GRAY)
        obs = np.array(obs_rgb).reshape(-1,self.width,self.height,3)[0]
        #print(obs.shape)        
        blood_window_gray = cv2.cvtColor(grab_screen(self.blood_window),cv2.COLOR_BGR2GRAY)
        stamina_window_gray = cv2.cvtColor(grab_screen(self.stamina_window),cv2.COLOR_BGR2GRAY)
        next_self_blood = self.self_blood_count(blood_window_gray)
        next_boss_blood = self.boss_blood_count(blood_window_gray)
        next_self_stamina = self.self_stamina_count(stamina_window_gray)
        next_boss_stamina = self.boss_stamina_count(stamina_window_gray)
        reward, done, stop, emergence_break = self.get_reward(self.boss_blood, next_boss_blood, self.self_blood, next_self_blood, 
                   self.boss_stamina, next_boss_stamina, self.self_stamina, next_self_stamina, 
                   self.stop, self.emergence_break,
                   self_blood_gamma=0.4, boss_blood_gamma=0.5, self_stamina_gamma=0.4, boss_stamina_gamma=0.6)
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        self.boss_stamina = next_boss_stamina
        #reward += loss_reward

        return (obs, reward, done, stop, emergence_break)
        

    def pause_game(self,paused):
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('start game')
                time.sleep(1)
            else:
                paused = True
                print('pause game')
                time.sleep(1)
        if paused:
            print('paused')
            while True:
                keys = key_check()
                # pauses game and can get annoying.
                if 'T' in keys:
                    if paused:
                        paused = False
                        print('start game')
                        time.sleep(1)
                        break
                    else:
                        paused = True
                        time.sleep(1)
        return paused

    def reset(self):
        restart()
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen,(self.width,self.height))
        obs_bgr = cv2.cvtColor(obs_resize,cv2.COLOR_BGR2RGB)
        obs_rgb = cv2.cvtColor(obs_bgr,cv2.COLOR_BGR2RGB)
        obs = np.array(obs_rgb).reshape(-1,self.width,self.height,3)[0]
        blood_window_gray = cv2.cvtColor(grab_screen(self.blood_window),cv2.COLOR_BGR2GRAY)
        stamina_window_gray = cv2.cvtColor(grab_screen(self.stamina_window),cv2.COLOR_BGR2GRAY)
        self.self_blood = self.self_blood_count(blood_window_gray)
        self.boss_blood = self.boss_blood_count(blood_window_gray)
        self.self_stamina = self.self_stamina_count(stamina_window_gray)
        self.boss_stamina = self.boss_stamina_count(stamina_window_gray)

        return obs

    

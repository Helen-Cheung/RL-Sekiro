'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-14 16:39:34
LastEditors: Zhang
LastEditTime: 2021-11-14 16:39:34
'''


import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
import time

import cv2
import mss
import numpy as np
import pyautogui
import sys
import os
from scipy.ndimage import maximum_filter

def find_templ(img, tpl):
    corr = cv2.matchTemplate( img, tpl, cv2.TM_CCOEFF_NORMED)
    max_match_map = np.max(corr) 
    
    if(max_match_map < 0.71): 
        return None
    
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    return (y, x)

def forward(arr, fy, fx, fps):
    delta = fps + 85
    arr = arr[fy+1, fx:fx+delta]
    return np.unique(arr).shape[0]

def screen_record():
    screenWidth, screenHeight = pyautogui.size()
    mon = (0, 0, screenWidth, screenHeight)

    sct = mss.mss()

    img_tp = cv2.imread("data/dinozever.jpg", cv2.IMREAD_GRAYSCALE)
    over = cv2.imread("data/over.jpg", cv2.IMREAD_GRAYSCALE)
    
    img = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGR2GRAY)
    
    coord = find_templ( img, img_tp )
    y = coord[0] + int(img_tp.shape[0] / 2) + 10
    x = coord[1] + int(img_tp.shape[1] / 2) + 30
    
    pyautogui.press('up')

    fps = 0
    while True:
        img = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGR2GRAY)
        
        if forward(img, y, x, fps) > 1:
            pyautogui.press('up')
            fps += 1

        if find_templ(img, over):
            fps = 0
            return 

time.sleep(3)
screen_record()

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:40:10 2021

@author: chen
"""
import os
import PIL.Image as im
from PIL import ImageDraw

url = "D:/yolo/yolov5_all-main/data/images/val_a_f"
box = (92, 32, 148, 109) # 裁剪區域

magnify = 3
rescale = 0.65 #缩小倍数
#實際區域 = magnify*rescale

tempbox = (box[0]*magnify, box[1]*magnify,box[2]*magnify, box[3]*magnify)
box3 = [(box[0],box[1]), (box[2],box[1]), (box[2],box[3]),(box[0], box[3]),(box[0],box[1])]
for pic in os.listdir(url):
     img = im.open(url+'/'+pic).convert('L')
     print(img.size)
     temp = img
     temp = temp.resize((img.size[0]*magnify, img.size[1]*magnify),im.ANTIALIAS)
     cropped = temp.crop(tempbox)
     croppedShape = cropped.size
     cropped = cropped.resize((int(croppedShape[0]*rescale), int(croppedShape[1]*rescale)), im.ANTIALIAS)
     box2 = (img.size[0]- (int(croppedShape[0]*rescale)),img.size[1]-int((croppedShape[1]*rescale)))#粘貼區域
     img.paste(cropped,box2)
     img = img.convert("RGB")
     draw = ImageDraw.Draw(img)
     draw.line(box3,width = 3,fill = 'red')
     img.save(url+'/'+"processed "+pic)

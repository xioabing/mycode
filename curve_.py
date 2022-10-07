from matplotlib import pyplot as plt
import numpy as np
import csv

from matplotlib.pyplot import MultipleLocator
# 用来正常显示中文标签
from torch import nn

plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False

# 定义两个空列表存放x,y轴数据点
x = []
y = []
z = []
w = []
with open("runs/results.csv",'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))  # 从csv读取的数据是str类型
#         print("x:",x)
        y.append(float(row[3]))
#         print("y:",y)
        # z.append(float(row[2]))
#         w.append(float(row[4]))


# 画折线图
# x = np.linspace(1, 12,12)
# y = [0.541,0.658,0.711,0.743,0.761,0.776,0.786,0.794,0.802,0.808,0.812,0.820]
# plt.figure(num=3, figsize=(5, 5))
# plt.ylim((0, 0.14))
# plt.scatter(x,y,s=2) #label='YOLOV5'
plt.plot(x,y,color='black',) #label='val_obj_loss'
plt.xlabel('Epochs',fontsize=18) #recall
plt.ylabel('Loss',fontsize=18)
my_y_ticks = np.arange(0, 0.14, 0.025)
plt.yticks(my_y_ticks)
# plt.title('演示从文件加载数据')


# plt.legend()
plt.show()
#
#coding:utf-8
import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, average_precision_score
#
# y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
#
# #   类型：np.array； gt标签
#
# y_scores = np.array(
#     [0.9, 0.75, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
# #   类型：np.array； 由大至小排序的阈值score,
#
# # 画曲线
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# plt.figure("P-R Curve")
# plt.title('Precision/Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.plot(recall, precision)
# plt.show()
#
# # 计算AP
# AP = average_precision_score(y_true, y_scores, average='macro', pos_label=1, sample_weight=None)
# print('AP:', AP)
from utils.autoanchor import kmean_anchors
kmean_anchors(dataset='data/mask_data_format.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)



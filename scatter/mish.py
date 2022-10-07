# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# e = math.e
#
#
# def tanh(x):
#     return (e ** x - e ** (-x)) / (e ** x + e ** (-x))
#
#
# def softplus(x):
#     return math.log(1 + pow(e, x))
#
#
# def mish(x):
#     return x * tanh(softplus(x))
#
#
# x = np.linspace(-5, 5, 1000)
# y = np.linspace(-5, 5, 1000)
# for i in range(1000):
#     y[i] = mish(x[i])
# plt.plot(x, y, color='black', linewidth=2, label='Mish')
# plt.legend()
# plt.savefig('mish.jpg')
# plt.show()

import numpy as np
import cv2 as cv

img = np.zeros((512,512,3),dtype=np.uint8)

cv.line(img,(0,0),(250,200),(0,0,255),5)
cv.circle(img,(255,255),50,(0,255,255),-1)
cv.putText(img,'opencv',(200,200),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),cv.LINE_AA,1)

winname = 'exp'
cv.namedWindow(winname)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyWindow(winname)
# from torch import nn
#
# from utils import autoanchor
#
# # 对数据集重新计算 anchors
# NEW = []
# if __name__=='__main__':
#     new_anchors = autoanchor.kmean_anchors('data/mask_data_format__.yaml', 12, 640, 5.0, 1000, True)
#     NEW.append(new_anchors)
#     print(NEW)
#


from utils.plots import plot_results,feature_visualization
# plot_results("runs/train/new/results.csv")
x = 'data/images/000060.jpg'
module_type = 13
stage = 4
feature_visualization(x,module_type,stage)
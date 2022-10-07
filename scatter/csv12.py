ls = open("D:/yolo/yolov5_all-main/mask_yolo_format/labels/result1.txt").readlines()
newTxt = ""
for line in ls:
    newTxt = newTxt + ",".join(line.split()) + "\n"
print(newTxt)

fo = open("D:/yolo/yolov5_all-main/mask_yolo_format/labels/new12.csv", "w")
fo.write(newTxt)
fo.close()


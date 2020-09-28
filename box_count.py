import cv2
import numpy as np
"""
box_top_s  堆疊高度(超過代表桶槽有堆疊)
box_space  區域個數
box_line   區域分割線，由左到右排序(X軸)
box_bottom 區域底線(Y軸)
box_top    obj.names[0]
box        obj.names[1]
"""
def box_count(classes, confidences, boxes, box_top_s = 150, box_space = 2, box_line = [850, 1400], box_bottom = 650):
    box_num = np.zeros((box_space))
    y_top_list = [[] for i in range(box_space)]
    y_top_box_list = [[] for i in range(box_space)]
    for classId, confidence, boundingBox in zip(classes.flatten(), confidences.flatten(), boxes):
        x_left, y_top, box_width, box_height = boundingBox
        #過濾目標區域以外的桶子
        if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom: # or confidence < 0.5
            continue
        #根據區域數量，單獨儲存box_top和box的y_top
        for i in range(box_space):
            if classId == 0: #box_top
                if x_left + box_width <= box_line[i]:
                    y_top_list[i].append(y_top)
                    box_num[i] = box_num[i] + 1
                    break
            elif classId == 1: #box
                if x_left + box_width <= box_line[i]:
                    y_top_box_list[i].append(y_top)
                    break
    #list to numpy array
    for i in range(box_space):
        y_top_list[i] = np.array(y_top_list[i])
        y_top_box_list[i] = np.array(y_top_box_list[i])
        
    print(y_top_list)
    print(y_top_box_list)

    #處理桶子堆疊問題
    for i in range(box_space):
        #判斷區域內是否有桶子
        if y_top_list[i].size != 0 and y_top_box_list[i].size != 0:
            diff_num = y_top_list[i][(y_top_list[i] - y_top_list[i].min()) > box_top_s] #第一層一噸桶
            #計算桶子數量
            #計算流程：將第一層的桶子過濾掉，將剩下的桶子數量乘以層高，再加上第一層的桶子數量
            if diff_num.size != 0:
                box_num[i] = (box_num[i] - diff_num.size) * 2 + diff_num.size
            else:
            #box_top都在同一層時，將box_top數量乘以層高，即一噸桶數量
                if y_top_list[i].size != 0:
                    if (y_top_box_list[i].max() - y_top_box_list[i].min()) > box_top_s:
                        box_num[i] = box_num[i] * 2
    return box_num

if __name__ == "__main__":
    net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
    net.setInputSize(512, 512)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    with open('obj.names', 'r') as f:
        names = f.read().rstrip('\n').split('\n')

    imagePath = "test.jpg"
    image = cv2.imread(imagePath)
    classes, confidences, boxes = net.detect(image, confThreshold=0.1, nmsThreshold=0.4)

    box_num = box_count(classes, confidences, boxes)
    print(box_num)
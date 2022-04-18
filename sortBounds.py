from functools import cmp_to_key
from math import cos, sin
import math
from xml.etree.ElementTree import PI
from PIL import Image
import glob
import os
from pathlib import Path
from cv2 import rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def cmp_x(w, z):
    return w.x - z.x

def cmp_y(w, z):
    return w[0].y - z[0].y

class Box:
    def __init__(self, x, y, w, h, l='?'):
        self.x = x
        self.y = y
        self.height = h
        self.width = w
        self.used = False
        self.lexeme = l

    def __repr__(self):
        return self.lexeme#'%dx%d (%d, %d)' % (self.width, self.height, self.x, self.y)
    
def sortBounds(boxes, y_tresh=20):
    rows = []
    # Create Rows
    for i in range(len(boxes)):         
        b = boxes[i]
        if b.used:
            continue
        row = [b]
        b.used = True
        for j in range(i, len(boxes)):
            b2 = boxes[j]
            bottom = b.y + b.height
            bottom2 = b2.y + b2.height
            if not b2.used and abs(bottom - bottom2) <= y_tresh:
                row.append(b2)
                b2.used = True
        rows.append(row)

    # Sort characters in row
    for r in rows:
        r.sort(key=cmp_to_key(cmp_x))
    # Sort order of rows
    rows.sort(key=cmp_to_key(cmp_y))
    
    return rows

def sort_single(lbl_name):
    txt = open('/var/www/s22/clara-g5/ocr_results/%s.txt' % lbl_name)
    txt_data = txt.read().splitlines()
    txt.close()

    yolo_path = '/var/www/s22/clara-g5/yolov5Letter/runs/detect/exp/'

    bb_file = open('%s%s/%s.txt' % (yolo_path, 'labels',  lbl_name))
    bb_data = bb_file.read().splitlines()
    bb_file.close()

    lbl_image = Image.open('%s%s.jpg' % (yolo_path, lbl_name))
    width = lbl_image.width
    height = lbl_image.height

    boxes = []

    for i, line in enumerate(bb_data):
        line_props = line.split(' ')

        x_center = float(line_props[1]) * width
        y_center = float(line_props[2]) * height
        w        = float(line_props[3]) * width
        h        = float(line_props[4]) * height

        x = x_center - w/2
        y = y_center - h/2

        b = Box(x, y, w, h, l=txt_data[i][0])
        boxes.append(b)

    #ans = sortBounds(boxes)
    ans = process_boxes(boxes)
    str_ans = []
    for r in ans:
        for c in r:
            str_ans.append(c.lexeme)
        str_ans.append('\n')
    str_ans = ''.join(str_ans)
    return str_ans
    

def sort_label_output(ipann_output_path):
    #CROP_PATH = os.path.join('./yolov5Letter/runs/', input_name
    for pth in glob.glob(ipann_output_path + '/*'):
        if os.path.isdir(pth) == False:
            #print('\n\n', pth)
            lbl_name = Path(pth).stem
            #print(lbl_name)
            sortd = sort_single(lbl_name)
#            print(sortd)
            out_file = open('/var/www/s22/clara-g5/ocr_results/%s.txt' % lbl_name, 'w+')
            out_file.write(sortd)
            out_file.close()

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]

def squareness_score(m):
    height = len(m)
    count = 0
    for r in m:
        diff = (height - len(r)) ** 2
        count += diff
    return math.sqrt(count)


def rotate_boxes(bs, theta, origin=[0,0]):
    rbs = []
    M = np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])             # Rotation Matrix
    T = np.array([[1, 0, 0], [0, 1, 0], [origin[0], origin[1], 1]])                                 # Translation Matrix
    Tt = T.transpose()
    for b in bs:
        v = np.array([b.x, b.y, 0])
        out = np.matmul(np.matmul(T, np.matmul(v, M)), Tt)
        rbs.append(Box(out[0], out[1], b.width, b.height, l=b.lexeme))
    return rbs

def draw_boxes(bs):
    fig, ax = plt.subplots()
    ax.plot([0, 400],[0, 400], linewidth=0)
    for box in bs:
        Y = 400 - box.y
        r = Rectangle((box.x, Y), box.width, box.height, edgecolor = 'green', facecolor = '#00000000',)
        ax.annotate(box.lexeme, (box.x + box.width/2, Y + box.height/2), )
        ax.add_patch(r)
    plt.show()

def process_boxes(bs):
    tested = []
    results = []
    for i in range(6):
        theta = i * math.pi/3
        t1 = rotate_boxes(bs, theta)
        ans = sortBounds(t1)
        tested.append(ans)
        results.append(squareness_score(ans))
    mindex = results.index(min(results))
    return tested[mindex]
    

def main():
    
    bs = [  Box(140.5, 347, 39, 52, '0'),
            Box(166, 191, 42, 54, '3'),
            Box(103.5, 349.5, 43, 55, '2'),
            Box(213, 339.5, 50, 55, '7'),
            Box(234.5, 262.5, 43, 55, '6'),
            Box(176, 344, 44, 48, '0'),
            Box(102.5, 291.5, 25, 23, '.'),
            Box(149.5, 115.5, 61, 61, 'Q'),
            Box(207, 111, 54, 64, 'H'),
            Box(205.5, 189.5, 39, 57, '7'),
            Box(143.5, 268.5, 59, 61, 'W'),
            Box(126.5, 194.5, 47, 55, '4'),
            Box(195, 265, 46, 64, '5')
        ]
    for i in range(6):
        theta = (math.pi/4) - i * math.pi/12
        t1 = rotate_boxes(bs, theta)
        #draw_boxes(t1)
        ans = sortBounds(t1)
        print('Theta:', theta, 'Score:', squareness_score(ans))
    print(ans)
    for r in ans:
        for c in r:
            print(c, end='')
        print()

if __name__ == '__main__':
    main()

'''
Matthew Kleitz, SUNY New Paltz, Spring 2022
'''
from pathlib import Path
import glob
import os
from config import PATH

def sort_all_from(src):
    '''
    Creates a txt file *-full.txt that lists the full txt output for the picture of a book shelf.
    After text files are produced for each label output, this function will will aggregate and sort them based on the bounding box data
    produced by yolov5.

    Input:
        - src   The name of a shelf image. EX: t1.jpg. Do not give the full path! Just something like NAME.jpg
    Output:
        - A text file with name $PROJECT_DIR/ocr_results/{src}-full.txt
    '''
    src_name = Path(src).stem

    bb_file = open('%syolov5/runs/detect/exp/labels/%s.txt' % (PATH(), src_name))
    bb_data = bb_file.read().splitlines()
    bb_file.close()
    
    print('Found %d bounding boxes!' % len(bb_data))

    index_order = [] # order that the outputted crops should appear in. so if = [(3,0),  (1,3),  ...] the first label should be third.
    
    # sort the bb data
    for i, line in enumerate(bb_data):
        line_props = line.split(' ')
        x_center = float(line_props[1])
        y_center = float(line_props[2])

        data = (i, x_center) # tuple to store the bb data
        
        if i == 0:
            index_order.append(data)
            continue
        
        # Sorting Part
        placed = False
        for j in range(len(index_order)):
            if x_center < index_order[j][1]:
                index_order.insert(j, data)
                placed = True
                break
        if not placed:
            index_order.append(data)

    print('Sorted %d labels.' % len(index_order))     
    full_file = open('%socr_results/%s-full.txt' % (PATH(), src_name), 'w+')
    
    # Create the text output
    for box in index_order:
        index = box[0] + 1
        index_str = str(index) if index != 1 else ''
        
        print('Box: (%d, %s)' % (index, src_name))
        
        try:
            # Read the individual label text file
            label_file = open('%socr_results/%s%s.txt' % (PATH(), src_name, index_str), mode='r')
            label_data = label_file.read()
            label_file.close()
            # Write that label data into the full file
            full_file.write(label_data)
            full_file.write('\n')
        except:
            print('Hmm... Could not find ocr txt file for: %s%s' % (src_name, index_str))

    full_file.close()

if __name__ == '__main__':
    print('Starting up... sorting label text files...')
    #sort_all_from('tA.jpg')
    PTH = PATH() + 'yolov5/runs/detect/exp/labels'

    for pth in glob.glob(PTH + '/*'):
        
        if os.path.isdir(pth) == False:
            name = Path(pth).stem
            img_name = name + '.jpg'
            sort_all_from(img_name)
            print('Sorted from %s!' % img_name)
    print('Finished.')


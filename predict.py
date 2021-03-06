'''
Matthew Kleitz, 2022
predict.py for the Spring 2022 SUNY New Paltz Projects Course
Book Label Image Processing Project
'''
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from ann.net import SPLIT, Net, PredictSymbolDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sortBounds import sort_label_output
from config import PATH

# Classes used in the neural network
correct_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '.']
index_to_class = {i:j for i, j in enumerate(correct_classes)}
class_to_index = {value:key for key,value in index_to_class.items()}

def load_net(weights):
    '''
    Loads the trained neural network. Weights are loaded from given file path 'weights'
    '''
    net = Net()
    checkpoint = torch.load(weights)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

# Transform used on inputted images for prediction
transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=128),
        ToTensorV2(),
    ]
)

def predict(net, img_tensor):
    '''
    Predicts the class of a single image.
    Inputs:
        net         - The Neural Network

        img_tensor  - The input image in tensor form
    Outputs:
        The predicted class (string).
    '''
    output = net(img_tensor.float())
    
    _, predicted = torch.max(output, 1)
    #__, pred2 = torch.topk(output, 3)
    #print(correct_classes[pred2[1]])
    #print('\t', predicted[0], predicted[1])
    predict_class = correct_classes[predicted[0]]

    return predict_class

def predict_folder(net, folder_path, output_file=None):
    '''
    Attempts to predict every image in a given folder path. 
    Inputs:
        folder_path     - The path containing the images to predict.

        output_file     - (Optional) The path to store the predicted text file.
                        - Default: {folder_path}/OUTPUT.txt
    Outputs:
        text file.
    '''
    test_folder = folder_path
    test_image_paths = []
    
    # Get the path of every image file in the test folder
    for data_path in sorted(glob.glob(test_folder + '/*'), key=os.path.getmtime):
        if '.jpg' in data_path or '.jpeg' in data_path:
            test_image_paths.append(data_path)
    
    # Create a dataset from the image paths
    test_dataset = PredictSymbolDataset(test_image_paths, transform)

    # Create a PyTorch data loader
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    
    # Initialize a data iterator and output variable
    dataiter = iter(test_loader)
    output = ''

    # Iterate over every image in the folder
    for i in tqdm(range(len(test_image_paths))):
        # Get the image data
        images = dataiter.next()
        # Get prediction
        res = predict(net, images.float())
        output += res + ' ' + test_image_paths[i] + '\n'

    if output_file == None:
        output_file = os.path.join(folder_path, 'OUTPUT.txt')

    # Save output
    file = open(output_file, 'w+')
    file.write(output)
    file.close()

def main(path, subdirs, out=None):
    '''
    Entry point. Loads the neural network and starts the prediction operation.
    '''
    print('Starting...')
    w_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights.pth')
    net = load_net(w_path)
    print('ANN Loaded. Processing...')

    sdir_count = 0
    if subdirs > 0:
        for data_path in glob.glob(path + '/*'):
            if os.path.isdir(data_path):
                out_path = (out[0] if out is not None else data_path) + '/' + data_path.split(SPLIT)[-1] + '.txt'
                sdir_count += 1
                predict_folder(net, data_path, out_path)
                print('Processed: %s' % out_path)
    else:
        sdir_count = 1
        predict_folder(net, path, out[0] if out is not None else None)
    
    if sdir_count == 0:
        print('\t[!Error!] It seems like the --path argument does not have any subdirectories! Nothing was processed!')
    else:
        print('Finished. Processed %d directories.' % sdir_count)

    print('Sorting output...')
    sort_label_output('%socr_results/' % PATH())
    print('Finished.')

if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description='~~~ IPANN (Image Processing Neural Network) ~~~\nSUNY New Paltz, 2022\nCLI Implemented by Matthew Kleitz')
    parser.add_argument('--path', type=str, required=True, help='Directory to process.')
    parser.add_argument('--out', type=str, nargs=1, help='Output path for detected characters. Text file format.')
    parser.add_argument('--subdirs', action='count', default=0, help='Process all folders given in --path argument.')
    args = parser.parse_args()
    print(args)

    main(args.path, args.subdirs, args.out)

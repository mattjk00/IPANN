# IPANN
## SUNY New Paltz, Spring 2022
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BzPH14UlSkKtqFjp0Wanaq3wRR0tlH5E)

This repository is home to the Character classification neural network, used in the CLARA project.

The image processing project for Spring 2022 is being worked on by Matthew Kleitz and Anas El Yousfi under the direction of Professor Hanh Pham

# Usage
Run the following command to classify the images in a folder. Replace /SOME_IMAGE_PATH/ with the folder containing jpg images.
```
python ./IPANN/predict.py --path /SOME_IMAGE_PATH/
```
To run predictions on every subdirectory in a given folder, add the --subdirs flag.

```
python ./IPANN/predict.py --path /SOME_IMAGE_PATH/ --subdirs
```
You can also specify an output directory using the --out argument.

```
python ./IPANN/predict.py --path /SOME_IMAGE_PATH/ --subdirs --out ./output/
```
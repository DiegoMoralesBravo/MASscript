import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

cudaFlag = torch.cuda.is_available()
print("CUDA is available:", cudaFlag)

def show_anns(anns, image, xList, yList, stepNumber):
    heightImage = image.shape[0]
    widthImage = image.shape[1]
    minHeight = heightImage
    minArea = anns[0]['area']
    maxArea = anns[0]['area']

    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse = True)

    for ann in sorted_anns:
        bbox = ann['bbox']
        area = ann['area']

        if bbox[0] < 10 and bbox[1] > (heightImage * 0.70) and bbox[2] > (widthImage * 0.7) and bbox[3] < (heightImage * 0.3):
            minHeight = bbox[1]

    sorted_anns = removeContainers(sorted_anns, minHeight, image)

    for ann in sorted_anns:
        bbox = ann['bbox']
        area = ann['area']

        if area < minArea:
            minArea = area
        if area > maxArea:
            maxArea = area

    step = (maxArea - minArea) / stepNumber

    for i in range(stepNumber):
        xList.append(str(round(minArea + (step * (i + 1)))))


    for ann in sorted_anns:
        bbox = ann['bbox']
        area = ann['area']
        cv2.rectangle(image, (bbox[0], bbox[1]),(bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0,0,255), thickness=1)
        fillDataChart(area, xList, yList)
    return image

def fillDataChart(area, xList, yList):
    for i, areaXlist in enumerate(xList):
        if area <= int(areaXlist):
            yList[i] += 1
            break

def removeContainers(anns, minHeight, image):
    minHeightAnns = [ann for ann in anns if ann['bbox'][1] < minHeight]
    withoutCointanerAnss = []

    for i in range(len(minHeightAnns)):
        counter = 0
        flag = True
        pxTainer = minHeightAnns[i]['bbox'][0]
        pyTainer = minHeightAnns[i]['bbox'][1]
        withTainer = minHeightAnns[i]['bbox'][2]
        heightTainer = minHeightAnns[i]['bbox'][3]

        for j in range(i + 1, len(minHeightAnns)):
            pxTained = minHeightAnns[j]['bbox'][0]
            pyTained = minHeightAnns[j]['bbox'][1]
            withTained = minHeightAnns[j]['bbox'][2]
            heightTained = minHeightAnns[j]['bbox'][3]

            if pxTained >= pxTainer - 5 and (pxTained + withTained) <= (pxTainer + withTainer + 5) and pyTained >= pyTainer - 5 and (pyTained + heightTained) <= (pyTainer + heightTainer + 5):
                counter += 1
                if counter == 2:
                    flag = False
                    break

        if flag:
            withoutCointanerAnss.append(minHeightAnns[i])

    return withoutCointanerAnss


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

if cudaFlag:
    flag = input('Use GPU available? Y/N')
    if flag == 'Y' or flag == 'y':
        device = "cuda"
    else:
        device = "cpu"
else:
    device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side= 44, 
    pred_iou_thresh = 0.85,
    stability_score_thresh = 0.97,
    crop_n_layers = 1,
    crop_n_points_downscale_factor = 2,
    min_mask_region_area= 1000,
)


stepNumber = 20
xList = []
yList = [0] * stepNumber

flag = True
while flag:
    try:
        image_path = input('Enter image path: ')
        image = cv2.imread(image_path)
        flag = False
    except Exception as e:
        print('Invalid image path', file=sys.stderr)
        print('Error al leer la imagen:', e)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
image = show_anns(masks, image, xList, yList, stepNumber)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
ax1.imshow(image)
ax2.bar(x=xList, height=yList)
plt.savefig('figure.png')

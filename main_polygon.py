import os
import argparse
from typing import Tuple
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np

import cv2
import skimage
import imantics

import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=r"C:\Users\Thanh\Downloads\husky-1.jpg")
    parser.add_argument("--num_polygon_points", type=int, default=20)
    parser.add_argument("--config_file", type=str, default="configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py")
    parser.add_argument("--model_path", type=str, default="checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth")

    args = parser.parse_args()

    return args



def parse_result(result: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes, masks = result

    bbox_result, mask_result = result
    bboxes = np.vstack(bbox_result)

    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

    labels = np.concatenate(labels)
    #print(labels, bboxes)
    indices = np.where(bboxes[:, 4] > 0.5)[0]
    #print(indices)
    if len(labels) == 0:
        bboxes = np.zeros([0, 5])
        masks = np.zeros([0, 0, 0])
    # draw segmentation masks
    else:
        masks = list(chain(*mask_result))
        if isinstance(masks[0], torch.Tensor):
            masks = torch.stack(masks, dim=0).detach().cpu().numpy()
        else:
            masks = np.stack(masks, axis=0)
        #print(masks.shape)
        # dummy bboxes
        labels = labels[indices]
        bboxes = bboxes[indices]
        masks = masks[indices]
        if bboxes[:, :4].sum() == 0:
            num_masks = len(bboxes)
            x_any = masks.any(axis=1)
            y_any = masks.any(axis=2)
            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                if len(x) > 0 and len(y) > 0:
                    bboxes[idx, :4] = np.array(
                        [x[0], y[0], x[-1] + 1, y[-1] + 1],
                        dtype=np.float32)

    return labels, bboxes, masks


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return pick


def visualize_image(image_path: str, labels: np.ndarray, bboxes: np.ndarray, masks: np.ndarray, num_polygon_points: np.ndarray) -> np.ndarray:
    img = cv2.imread(image_path)
    #print(img.dtype)
    color_mask = np.zeros_like(img, dtype=np.uint8)
    polygon_list = []
    for label, bbox, mask in zip(labels, bboxes, masks):
        #print(label.shape, bbox.shape, mask.shape)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        #color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
        polygon = imantics.Mask(array=mask).polygons()
        #img = polygon.draw(image=img, color=(100, 100, 0))
        points = polygon.points
        index = np.linspace(0, points[0].shape[0], num_polygon_points, endpoint=False, dtype=np.int32)
        #print("Points: {}".format(points[0].shape))
        r = points[0].T[1]
        c = points[0].T[0]
        rr, cc = skimage.draw.polygon(r, c)
        color_mask[rr, cc] = np.array(color, dtype=np.uint8)
        points = points[0][index].reshape(-1, 1, 2)
        polygon_list.append(points)
        img = cv2.polylines(img, [points], True, color, 3)
    color_mask = color_mask.astype(np.uint8)
    img = np.where(color_mask > 0, cv2.addWeighted(img, 0.4, color_mask, 0.6, 0), img)
    polygons = np.stack(polygon_list, axis=0)
    polygons = np.squeeze(polygons, axis=2)
    return img, polygons


if __name__ == "__main__":
    args = get_args()

    image_path = args.image_path
    num_polygon_points = args.num_polygon_points
    config_file = args.config_file
    model_path = args.model_path

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = init_detector(config=config_file, checkpoint=model_path, device=device)

    result = inference_detector(model=model, imgs=image_path)

    #model.show_result(image_path, result, show=True)

    labels, bboxes, masks = parse_result(result=result)

    indices = non_max_suppression_fast(boxes=bboxes, overlapThresh=0.7)

    labels = labels[indices]
    bboxes = bboxes[indices]
    masks = masks[indices]

    #print("Masks: {}".format(masks.shape))

    img, polygons = visualize_image(image_path=image_path, labels=labels, bboxes=bboxes, masks=masks, num_polygon_points=num_polygon_points)

    print("Polygons: {}".format(polygons.shape))

    plt.imshow(img[:, :, ::-1])
    plt.show()

import numpy as np

# NOTE/ This portion of code is lifted from Adrian Rosebrock's NMS
# with probabilities example. See www.pyimagesearch.com for more details.

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	'Select picks from candidate bounding boxes based on input probabilities'
    # If there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# If the bounding boxes are integers convert them to floats for division
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# Initialize the list of picked indices
	pick = []

	# Grab coordinates of bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# Compute the area of the bounding boxes and grab the indices to sort
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# If probabilities are provided, sort on them
	if probs is not None:
		idxs = probs

	# Sort the indexes
	idxs = np.argsort(idxs)

	# Loop while indexers remain in the indexes list
	while len(idxs) > 0:
		# Grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# Find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# Compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# Compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# Delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# Return only bounding boxes that were picked
	return boxes[pick].astype("int")
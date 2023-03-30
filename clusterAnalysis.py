import os
import numpy as np


def extract_boxes_wh(data):
    """
    Extract width and height from description files with bounding boxes.

    param: data - path to obj.data
    return: bounding boxes with only width and height

    Note: only the train folder is taken from obj.data
    """
    with open(data, 'r') as f:
        f.readline()
        data_dir = f.readline().split()[2]
        f.readline()
        labels_dir = f.readline().split()[2]
        f.readline()
        file_format = f.readline().split()[2]

    class2tag = {}
    with open(labels_dir, 'r') as f:
        for line in f:
            (val, key) = line.split()
            class2tag[key] = val

    box_paths = []
    for tag in class2tag:
        for file in os.listdir(data_dir + '/' + tag):
            if file.endswith('.' + file_format):
                box_paths.append(data_dir + '/' + tag + '/' + file)

    boxes = []
    for file in box_paths:
        with open(file) as f:
            for obj in f:
                param_list = list(map(float, obj.split()))
                boxes.append(param_list[3:])

    return np.array(boxes)


def inersection_over_union(boxes_wh, clusters):
    w1, h1 = boxes_wh

    intersection_area = np.minimum(w1, clusters[:, 0]) * np.minimum(h1, clusters[:, 1])
    union_area = (w1 * h1) + (clusters[:, 0] * clusters[:, 1]) - intersection_area
    iou = intersection_area / union_area

    return iou


def average_intersection_over_union(boxes_wh, clusters):
    return np.mean([np.max(inersection_over_union(boxes_wh[i], clusters)) for i in range(boxes_wh.shape[0])])


def kmeans(boxes_wh, n_cluster):
    """
    k-means clustering with custom distance

    param: boxes_wh - width and height bounding boxes
    param: n_cluster - number of clusters

    return: bounding boxes clusters (width and height)
    """
    num_boxes = boxes_wh.shape[0]
    distances = np.empty((num_boxes, n_cluster))
    last_clusters = np.empty(num_boxes)
    clusters = boxes_wh[np.random.choice(num_boxes, n_cluster, replace=False)]

    while True:
        for i in range(num_boxes):
            distances[i] = 1 - inersection_over_union(boxes_wh[i], clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if last_clusters.all() == nearest_clusters.all():
            break

        for i in range(n_cluster):
            clusters[i] = np.mean(boxes_wh[nearest_clusters == i], axis=0)
        last_clusters = nearest_clusters

    return clusters


def main():
    boxes_wh = extract_boxes_wh(data='./data/obj.data')
    clusters = kmeans(boxes_wh=boxes_wh, n_cluster=5)
    print(f'average IOU: {average_intersection_over_union(clusters, boxes_wh):.3f}')
    print('anchor boxes:\n', clusters)


if __name__ == "__main__":
    main()

from disable_warnings import *
from tensorflow.python.platform.tf_logging import info
import os
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import cv2

BATCH_SIZE = 64
im_width = 224
im_height = 224

model = tf.saved_model.load("./tmp/mymodel/1")
infer = model.signatures["serving_default"]

def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=5):
    """Draws bounding boxes on image (numpy array).
  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: a list of strings for each bounding box.
  Raises:
    ValueError: if boxes is not a [N, 4] array
  """

    draw_bounding_boxes_on_image(image, boxes, color, thickness)

    return image


def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=5):
    """Draws bounding boxes on image.
  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
                           
  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3], boxes[i, 2], color[i], thickness)


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5):
    """Adds a bounding box to an image.
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
  """
    im_width = image.shape[1]
    im_height = image.shape[0]
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)


# @title Visualization Utilities [RUN ME]
"""
This cell contains helper functions used for visualization
and downloads only. You can skip reading it. There is very
little useful Keras/Tensorflow code here.
"""

# Matplotlib config
plt.rc("image", cmap="gray")
plt.rc("grid", linewidth=0)
plt.rc("xtick", top=False, bottom=False, labelsize="large")
plt.rc("ytick", left=False, right=False, labelsize="large")
plt.rc("axes", facecolor="F8F8F8", titlesize="large", edgecolor="white")
plt.rc("text", color="a8151a")
plt.rc("figure", facecolor="F0F0F0")  # Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


# utility to display a row of digits with their predictions
def display_digits_with_boxes(images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False):

    n = len(images)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if len(pred_bboxes) > i:
            bbox = pred_bboxes[i]
            bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1], bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        if len(bboxes) > i:
            bbox = bboxes[i]
            if bboxes_normalized == True:
                bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1], bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        img_to_draw = draw_bounding_boxes_on_image_array(image=images[i], boxes=np.asarray(bboxes_to_plot), color=[(255, 0, 0), (0, 255, 0)])
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if iou[i][0] < iou_threshold:
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (iou[i][0]), color=color, transform=ax.transAxes)


# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_" + metric_name], color="green", label="val_" + metric_name)


"""
Resizes image to (224, 224), normalizes image and translates and normalizes bounding boxes.
"""


def read_image_tfds(image, bbox):
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(image, (224, 224,))

    image = image / 127.5
    image -= 1

    return image, [bbox[0] / factor_x, bbox[1] / factor_y, bbox[2] / factor_x, bbox[3] / factor_y]


"""
Helper function to read resized images, bounding boxes and their original shapes.
Resizes image to (224, 224), normalizes image and translates and normalizes bounding boxes.
"""


def read_image_with_shape(image, bbox):
    original_image = image
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(image, (224, 224,))

    image = image / 127.5
    image -= 1

    return original_image, image, [bbox[0] / factor_x, bbox[1] / factor_y, bbox[2] / factor_x, bbox[3] / factor_y]


"""
Reads image and denormalized bounding boxes
"""


def read_image_tfds_with_original_bbox(data):
    image = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)

    return (
        image,
        [bbox[1] * tf.cast(shape[1], tf.float32), bbox[0] * tf.cast(shape[0], tf.float32), bbox[3] * tf.cast(shape[1], tf.float32), bbox[2] * tf.cast(shape[0], tf.float32)],
    )  # [bbox[0] * factor_x , (bbox[1] * factor_y), (bbox[2] * factor_x), (bbox[3] * factor_y)]


"""
Convert dataset to numpy arrays of images and boxes.
"""


def dataset_to_numpy_util(dataset, batch_size=0, N=0):

    # eager execution: loop through datasets normally
    take_dataset = dataset.shuffle(1024)

    if batch_size > 0:
        take_dataset = take_dataset.batch(batch_size)

    if N > 0:
        take_dataset = take_dataset.take(N)

    if tf.executing_eagerly():
        ds_images, ds_bboxes = [], []
        for images, bboxes in take_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())

    return (np.array(ds_images), np.array(ds_bboxes))


"""
Convert dataset to numpy arrays of original images, resized and normalized images and bounding boxes.
This is used for plotting the original images with true and predicted bounding boxes.
"""


def dataset_to_numpy_with_original_bboxes_util(dataset, batch_size=0, N=0):

    normalized_dataset = dataset.map(read_image_with_shape)
    if batch_size > 0:
        normalized_dataset = normalized_dataset.batch(batch_size)

    if N > 0:
        normalized_dataset = normalized_dataset.take(N)

    if tf.executing_eagerly():
        ds_original_images, ds_images, ds_bboxes = [], [], []
        for original_images, images, bboxes in normalized_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())
            ds_original_images.append(original_images.numpy())

    return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)  # , np.array(ds_normalized_images), np.array(ds_normalized_bboxes)


"""
Loads and maps the training split of the dataset. It used map function to reverse the normalization done on the bounding boxes in the dataset.
This will generate the dataset prepared for visualization
"""


def get_visualization_training_dataset():
    dataset = tfds.load("caltech_birds2010", split="train", data_dir="./data", download=False, builder_kwargs={"version": "0.1.1"})
    print(info)
    visualization_training_dataset = dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls=16)
    return visualization_training_dataset


"""
Loads and maps the validation split of the dataset. It used map function to reverse the normalization done on the bounding boxes in the dataset.
This will generate the dataset prepared for visualization
"""


def get_visualization_validation_dataset():
    dataset, info = tfds.load("caltech_birds2010", split="test", with_info=True, data_dir="./data", download=False)
    visualization_validation_dataset = dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls=16)
    return visualization_validation_dataset


"""
Loads and maps the training split of the dataset using the map function for resizing, image normalization and bounding box translation.
"""


def get_training_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(-1)
    return dataset


"""
Loads and maps the validation split of the dataset using the map function for resizing, image normalization and bounding box translation.
"""


def get_validation_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


# instantiate the datasets
visualization_training_dataset = get_visualization_training_dataset()
visualization_validation_dataset = get_visualization_validation_dataset()

training_dataset = get_training_dataset(visualization_training_dataset)
validation_dataset = get_validation_dataset(visualization_validation_dataset)


(visualization_training_images, visualization_training_bboxes) = dataset_to_numpy_util(visualization_training_dataset, N=10)
(visualization_validation_images, visualization_validation_bboxes)= dataset_to_numpy_util(visualization_validation_dataset, N=10)
validation_steps = 3033//BATCH_SIZE

'''
Calulcates and returns list of iou scores for all images in the test set
'''
def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  np.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)

    #Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(xmin_pred, xmin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    #Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred + 1) * (ymax_pred - ymin_pred + 1)
    true_box_area = (xmax_true - xmin_true + 1) * (ymax_true - ymin_true + 1)

    #Calculates overlap area and union area.
    overlap_area = np.maximum((xmax_overlap - xmin_overlap) + 1,0)  * np.maximum((ymax_overlap - ymin_overlap) + 1, 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    #Updates iou score
    iou = overlap_area / union_area

    return iou

#Makes predictions
original_images, normalized_images, normalized_bboxes = dataset_to_numpy_with_original_bboxes_util(visualization_validation_dataset, N=500)
# predicted_bboxes = model.predict(normalized_images, batch_size=32)
predictions = infer(tf.convert_to_tensor(normalized_images[0]))
print(predictions)

#Calculates IOU and reports true positives and false positives based on IOU threshold
iou = intersection_over_union(predicted_bboxes, normalized_bboxes)
iou_threshold = 0.7

print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))
from disable_warnings import *
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

BATCH_SIZE = 64
model = tf.keras.models.load_model("birds.h5")


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
    )


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

    return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)


"""
Loads and maps the validation split of the dataset. It used map function to reverse the normalization done on the bounding boxes in the dataset.
This will generate the dataset prepared for visualization
"""
def get_visualization_validation_dataset():
    dataset = tfds.load("caltech_birds2010", split="test", try_gcs=True, data_dir="./data", download=False)
    visualization_validation_dataset = dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls=16)
    return visualization_validation_dataset


"""
Loads and maps the validation split of the dataset using the map function for resizing, image normalization and bounding box translation.
"""
def get_validation_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


"""
Calulcates and returns list of iou scores for all images in the test set
"""
def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)

    # Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(xmin_pred, xmin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    # Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred + 1) * (ymax_pred - ymin_pred + 1)
    true_box_area = (xmax_true - xmin_true + 1) * (ymax_true - ymin_true + 1)

    # Calculates overlap area and union area.
    overlap_area = np.maximum((xmax_overlap - xmin_overlap) + 1, 0) * np.maximum((ymax_overlap - ymin_overlap) + 1, 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    # Updates iou score
    iou = overlap_area / union_area

    return iou



# instantiate the datasets
visualization_validation_dataset = get_visualization_validation_dataset()

original_images, normalized_images, normalized_bboxes = dataset_to_numpy_with_original_bboxes_util(visualization_validation_dataset, N=500)
predicted_bboxes = model.predict(normalized_images, batch_size=32)
iou = intersection_over_union(predicted_bboxes, normalized_bboxes)
iou_threshold = 0.7

beating_threshold = (iou >= iou_threshold).sum()
under_threshold = (iou < iou_threshold).sum()

grade = beating_threshold * 100 / (beating_threshold + under_threshold)
print(grade)
PASSING_GRADE = 50
if grade > PASSING_GRADE:
    print("You Passed")
else:
    print("You Failed")

from disable_warnings import *
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tools import send_feedback
import converter


if __name__ == "__main__":
    try:
        part_id = sys.argv[2]
    except IndexError:
        send_feedback(0.0, "Missing partId.", err=True)
    else:
        if part_id != "wNSsr":
            send_feedback(0.0, "Invalid partId.", err=True)

try:
    learner_model = tf.saved_model.load("./tmp/mymodel/1")
except:
    send_feedback(0.0, "Your model could not be loaded. Make sure the zip file has the correct contents.", err=True)




# for images, labels in test_batches:
#     try:
#         predictions = infer(images)["output_1"]
#     except:
#         send_feedback(0.0, "There was an issue with your model that prevented inference.")
#     eval_accuracy(labels, predictions)

# score = (eval_accuracy.result() * 100).numpy()

# if score > 60:
#     send_feedback(1.0, "Congratulations! Your model achieved the desired level of accuracy.")
# else:
#     send_feedback(0.0, "Your model has an accuracy lower than 0.6.")

import os
import shutil
from zipfile import ZipFile
from tools import send_feedback

submission_dir = "/shared/submission/"
sub_destination = "/grader/mymodel.zip"

if os.getenv("LOCALGRADER"):
    submission_dir = "./submission/"
    sub_destination = "./mymodel.zip"

learner_file = None
for file in os.listdir(submission_dir):
    if file.endswith(".zip"):
        learner_file = file

if learner_file is None:
    send_feedback(0.0, "No .zip was found in the submission directory.", err=True)

sub_source = submission_dir + learner_file
shutil.copyfile(sub_source, sub_destination)

saved_model_path = "./mymodel.zip"

with ZipFile(saved_model_path, "r") as zipObj:
    zipObj.extractall("./")

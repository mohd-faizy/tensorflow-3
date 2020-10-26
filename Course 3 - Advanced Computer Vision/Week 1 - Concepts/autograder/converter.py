import os
import shutil
from tools import send_feedback

submission_dir = "/shared/submission/"
sub_destination = "/grader/birds.h5"

if os.getenv("LOCALGRADER"):
    submission_dir = "./submission/"
    sub_destination = "./birds.h5"

learner_file = None
for file in os.listdir(submission_dir):
    if file.endswith(".h5"):
        learner_file = file

if learner_file is None:
    send_feedback(0.0, "No h5 file was found in the submission directory.", err=True)

sub_source = submission_dir + learner_file
shutil.copyfile(sub_source, sub_destination)

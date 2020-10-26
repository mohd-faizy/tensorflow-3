from disable_warnings import *
import sys
from tools import send_feedback


def grader():
    PASSING_GRADE = 50
    if grade > PASSING_GRADE:
        send_feedback(1.0, f"Congratulations!\nYour model achieved an iou score greater than 0.7 for {grade}% of the images.")
    else:
        send_feedback(0.0, f"Your model achieved an iou score greater than 0.7 for {grade}% of the images.\nAt least {PASSING_GRADE}% is required to pass.")
        

if __name__ == "__main__":
    try:
        part_id = sys.argv[2]
    except IndexError:
        send_feedback(0.0, "Missing partId.", err=True)
    else:
        if part_id != "wNSsr":
            send_feedback(0.0, "Invalid partId.", err=True)
        import converter
        from utils import grade
        grader()

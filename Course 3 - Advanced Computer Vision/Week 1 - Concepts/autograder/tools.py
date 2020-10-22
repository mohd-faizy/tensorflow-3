import sys
import json

def send_feedback(score, msg):
    post = {"fractionalScore": score, "feedback": msg}
    print(json.dumps(post))
    exit()

def print_stderr(error_msg):
    print(str(error_msg), file=sys.stderr)
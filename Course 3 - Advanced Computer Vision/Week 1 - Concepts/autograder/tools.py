import sys
import json

def send_feedback(score, msg, err=False):
    post = {"fractionalScore": score, "feedback": msg}
    print(json.dumps(post))
    if error:
        print(str(msg), file=sys.stderr)
    exit()
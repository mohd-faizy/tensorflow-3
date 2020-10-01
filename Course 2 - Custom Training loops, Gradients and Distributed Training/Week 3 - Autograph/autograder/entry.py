import sys
from disable_warnings import *
from tools import print_stderr, send_feedback
from grader import (Test_map_fn, Test_set_adam_optimizer, 
                    Test_set_sparse_cat_crossentropy_loss, Test_set_sparse_cat_crossentropy_accuracy,
                    Test_prepare_dataset, Test_train_one_step, Test_train)



def run_grader(part_id):
    
    graded_funcs = {
        "1": Test_map_fn, 
        "2": Test_set_adam_optimizer,
        "3": Test_set_sparse_cat_crossentropy_loss,
        "4": Test_set_sparse_cat_crossentropy_accuracy,
        "5": Test_prepare_dataset,
        "6": Test_train_one_step,
        "7": Test_train
    }

    g_func = graded_funcs.get(part_id)
    if g_func is None:
        print_stderr("The partID provided did not correspond to any graded function.")
        return

    failed_cases, num_cases = g_func()
    score = 1.0 - len(failed_cases) / num_cases
    if failed_cases:
        failed_msg = ""
        for failed_case in failed_cases:
            failed_msg += f"Failed {failed_case.get('name')}.\nExpected:\n{failed_case.get('expected')},\nbut got:\n{failed_case.get('got')}.\n\n"
        
        send_feedback(score, failed_msg)
    else:
        send_feedback(score, "All tests passed! Congratulations!")
    

if __name__ == "__main__":
    try:
        part_id = sys.argv[2]
    except IndexError:
        print_stderr("Missing partId. Required to continue.")
        send_feedback(0.0, "Missing partId.")
    else:
        run_grader(part_id)

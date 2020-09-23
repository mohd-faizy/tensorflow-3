import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import EagerTensor
import compiled
import learner_mod
import solution_mod
from tools import send_feedback, table_testing_assert

def get_failed_cases(test_cases):
    
    failed_cases = []
    
    for test_case in test_cases:
        name = test_case.get("name")
        got = test_case.get("got")
        expected = test_case.get("expected")
        if None in (name, got, expected):
            raise Exception("malformed test case")
        try:
            assert got == expected
        except:
            failed_cases.append({"name": name, "expected": expected, "got": got})
    
    return failed_cases

def Test_tf_constant():
    x = np.arange(41, 50)
    y = np.arange(71, 80)
    
    target = learner_mod.tf_constant
    solution = tf.constant
    
    result1 = target(x)
    result2 = target(y)

    if type(result1) != EagerTensor:
        failed_cases = [{"name": "type_check", 
                         "expected": EagerTensor, 
                         "got": type(result1)}]
        
        return failed_cases, 1
    
    else:
        test_cases = [
            {
                "name": "type_check",
                "got": type(result1),
                "expected": EagerTensor,
                "error_message": 'result has an incorrect type.'
            },
            {
                "name": "shape_check",
                "got": result1.shape,
                "expected": (len(x),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": f'output_equality_check_1 -- hidden test output: {x}, learner output: {result1.numpy()}',
                "got": np.array_equal(result1.numpy(), x),
                "expected": True,
                "error_message": f'output of the hidden test is {result1.numpy()} while you got {x}'
            },
            {
                "name": "output_array_check_2",
                "got": np.array_equal(result2.numpy(), y),
                "expected": True,
                "error_message": f'output of the hidden test is {result2.numpy()} while you got {y}'
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)

def Test_tf_square():
    pass

def Test_tf_reshape():
    pass

def Test_tf_cast():
    pass

def Test_tf_multiply():
    pass

def Test_tf_add():
    pass

def Test_tf_gradient_tape():
    pass

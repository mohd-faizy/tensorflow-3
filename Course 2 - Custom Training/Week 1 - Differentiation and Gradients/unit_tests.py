import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import EagerTensor


def test_loop(test_cases):
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            assert test_case["result"] == test_case["expected"]
            success += 1
    
        except:
            fails += 1
            print(f'{test_case["name"]}: {test_case["error_message"]}\nExpected: {test_case["expected"]}\nResult: {test_case["result"]}\n')

    if fails == 0:
        print("\033[92m All tests passed")

    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        raise Exception(test_case["error_message"])


def test_tf_constant(tf_constant):

    x = np.arange(1, 10)
    y = np.arange(11, 20)
    
    target = tf_constant
    
    result1 = target(x)
    result2 = target(y)
    
    expected_result1 = x
    expected_result2 = y
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(result1),
            "expected": EagerTensor,
            "error_message": 'result has an incorrect type.'
        },
        {
            "name": "shape_check",
            "result": result1.shape,
            "expected": (9,),
            "error_message": "output shape is incorrect"
        },
        {
            "name": "dtype_check",
            "result": result1.dtype,
            "expected": np.int64,
            "error_message": "output dtype is incorrect"
        },
        {
            "name": "output_check_1",
            "result": np.array_equal(result1.numpy(), expected_result1),
            "expected": True,
            "error_message": "output array is incorrect"
        },
        {
            "name": "output_check_2",
            "result": np.array_equal(result2.numpy(), expected_result2),
            "expected": True,
            "error_message": "output array is incorrect"
        },
    ]
    
    test_loop(test_cases)



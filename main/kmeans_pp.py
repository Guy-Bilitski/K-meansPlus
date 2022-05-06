import sys
import pandas as pd
import numpy as np
import kmeans_class_utils as class_utils

def main():
    try:
        args = load_args()
        print(args)
    except Exception as ex:
        print(ex)
        return


def load_args():
    """ returns a dict with all args that mentioned in ENV class and inputted """
    inp = sys.argv
    print (inp)
    args = {}
    if len(inp) < 5 or len(inp) > 6:
        raise Exception("Invalid Input!")
    try:
        args[class_utils.Env.k] = int(inp[1])
        if args[class_utils.Env.k] < 1:
            raise

        args[class_utils.Env.input_file1] = inp[-2]
        args[class_utils.Env.input_file2] = inp[-1]
        epsilon = float(inp[-3])
        if epsilon <= 0:
            raise
        else:
            args[class_utils.Env.epsilon] = inp[-3]

        if len(inp) == 6: # Checking maxiter exists
            maxiter = int(inp[2])
            if maxiter <= 0: # Validating maxiter value
                raise
            else:
                args[class_utils.Env.maxiter] = maxiter # If exists, added to args.
        
    except Exception:
        raise Exception("Invalid Input!")
    
    return args


main()
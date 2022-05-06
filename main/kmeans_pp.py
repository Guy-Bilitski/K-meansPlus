from operator import indexOf
import sys
import pandas as pd
import numpy as np
from scipy.misc import central_diff_weights

class Env:
    """ Class for global variables used in the system """
    k = "k"
    epsilon = "epsilon"
    input_file1 = "input_file1"
    input_file2 = "input_file2"
    maxiter = "maxiter"

def main():
    try:
        args = load_args()
        input_data_frame = get_df(args.get(Env.input_file1),
                                  args.get(Env.input_file2))
        np_array = pd.DataFrame.to_numpy(input_data_frame, dtype=float)
        get_centriods(np_array, args.get(Env.k))
        # Here comes the integration
    except Exception as ex:
        print(ex)
        return


def get_centriods(np_array, k):
    np.random.seed(0)
    n = np_array.shape[0]
    c1 = np_array[np.random.choice(n)]
    centroids = [c1]
    weighted_p = [0]*100

    for _ in range(k - 1):
        for j in range(n):
            x = np_array[j]
            closest_c = min(centroids, key=lambda c: np.linalg.norm(x - c))
            weighted_p[j] = np.linalg.norm(x-closest_c)
        distance_sum = sum(weighted_p)
        weighted_p = list(map(lambda x: x / distance_sum, weighted_p))
        new_cent_index = np.random.choice(n, p=weighted_p)
        centroids.append(np_array[new_cent_index])
    
    print(centroids)


def get_df(input_file1, input_file2):
    df1 = pd.read_csv(input_file1, header=None, dtype=float)
    df2 = pd.read_csv(input_file2, header=None, dtype=float)
    final_df = pd.merge(df1, df2, how='inner', on=0)
    return final_df.iloc[:, 1:]


def load_args():
    """ returns a dict with all args that mentioned in ENV class and inputted """
    inp = sys.argv
    print (inp)
    args = {}
    if len(inp) < 5 or len(inp) > 6:
        raise Exception("Invalid Input!")
    try:
        args[Env.k] = int(inp[1])
        if args[Env.k] < 1:
            raise

        args[Env.input_file1] = inp[-2]
        args[Env.input_file2] = inp[-1]
        epsilon = float(inp[-3])
        if epsilon <= 0:
            raise
        else:
            args[Env.epsilon] = inp[-3]

        if len(inp) == 6: # Checking maxiter exists
            maxiter = int(inp[2])
            if maxiter <= 0: # Validating maxiter value
                raise
            else:
                args[Env.maxiter] = maxiter # If exists, added to args.
        
    except Exception:
        raise Exception("Invalid Input!")
    
    return args

main()
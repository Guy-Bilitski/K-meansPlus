from mimetypes import init
from operator import indexOf
import sys
import pandas as pd
import numpy as np
import time
import mykmeanssp

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
        data_points = pd.DataFrame.to_numpy(input_data_frame, dtype=float)
        print(data_points)
        initial_centroids = get_centriods(data_points, args.get(Env.k))
        data_points = [c.tolist() for c in data_points]
        # Here come s the integration
        mykmeanssp.getKmeans(data_points, initial_centroids, args.get(Env.maxiter, -1), args.get(Env.epsilon))
    except Exception as ex:
        print(ex)
        return
    


def get_centriods(np_array, k):
    np.random.seed(0)
    n = np_array.shape[0]
    c1 = np_array[np.random.choice(n)]
    centroids = [c1]
    weighted_p = np.zeros(n, dtype=float)

    for _ in range(k - 1):
        for j in range(n):
            x = np_array[j]
            weighted_p[j] = (min(np.linalg.norm(x - c) for c in centroids))**2
        distance_sum = sum(weighted_p)
        np.divide(weighted_p, distance_sum, out=weighted_p)
        new_cent_index = np.random.choice(n, p=weighted_p)
        centroids.append(np_array[new_cent_index])
    centroids = [c.tolist() for c in centroids]
    for c in centroids:
        c.append(1.0)
    return centroids


def get_df(input_file1, input_file2):
    df1 = pd.read_csv(input_file1, header=None, dtype=float)
    df2 = pd.read_csv(input_file2, header=None, dtype=float)
    final_df = pd.merge(df1, df2, how='inner', on=0, copy=False, sort=True)
    # final_df.drop(columns=0, inplace=True)
    return final_df


def load_args():
    """ returns a dict with all args that mentioned in ENV class and inputted """
    inp = sys.argv

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
            args[Env.epsilon] = epsilon

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
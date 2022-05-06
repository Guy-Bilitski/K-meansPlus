class Env:
    """ Class for global variables used in the system """
    k = "k"
    epsilon = "epsilon"
    input_file1 = "input_file1"
    input_file2 = "input_file2"
    maxiter = "maxiter"

class Centroid:
    """
        Class represents a centroid of k-means algorithm 
        A cenroid is represented by the sum of all vectors assigned to the centroid
        and the number of vectors as count
    """
    def __init__(self, vector: float = None):
        if vector is None:
            self.count = 0
            self.vec_sum = []
        else:
            self.vec_sum = vector
            self.count = 1
    
    def __str__(self) -> str:

        return str([self[i] for i in range(len(self))])

    
    def __len__(self) -> int:
        return len(self.vec_sum)

    def __getitem__(self, i: int) -> float:
        return self.vec_sum[i]/self.count

    def add_vector(self, vector: list):
        if self.count == 0:
            self.vec_sum = vector
        else:
            for i in range(len(self.vec_sum)):
                self.vec_sum[i] += vector[i]
                
        self.count += 1
    
    def distance(self, vector) -> float:
        dist = 0
        for i in range(len(vector)):
            dist += ( self[i] - vector[i] )**2
        return dist**0.5
    
    def max_distance(centroids1, centroids2):
        maxd = float('-inf')
        for i in range(len(centroids1)):
            maxd = max(maxd, centroids1[i].distance(centroids2[i]))
        return maxd
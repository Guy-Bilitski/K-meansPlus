from sklearn.datasets import load_iris
from sklearn.cluster import KMeans



def main():
    data = load_iris().get('data')
    interias = []
    for k in range(2,11):
        print(f'calctulating kmeans with {k} clusters')
        kmeans = KMeans(random_state=0, n_clusters=k).fit(data)
        data_point_to_centroid = kmeans.predict(data)
        interias.append(get_interia(kmeans.cluster_centers_, data, data_point_to_centroid))
    print(interias)


def get_interia(centroids, data_points, data_point_to_centroid):
    sum = 0
    for i in range(len(data_points)):
        data_point_sum = 0
        current_data_point = data_points[i]
        current_centroid = centroids[data_point_to_centroid[i]]
        #print(f'index: {i}, current data point: {current_data_point} and current cent: {current_centroid}')
        for j in range(len(current_data_point)):
            data_point_sum += (current_data_point[j] - current_centroid[j])**2
        #print(f'data point {i} sum is {data_point_sum}')
        sum += data_point_sum
    
    return sum
    
    

if __name__ == '__main__':
    main()
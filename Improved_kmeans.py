# Hrishikesh Tiwari, Sunday . Nov 10 . 2024 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def euclid_distance(point1 , point2):
    if(point1.shape != point2.shape):
        raise ValueError("The shapes of the vectors are not same!")
    else:
        distt = 0 
        for i in range(point1.shape[0]):
            distt += (point1[i] - point2[i])**2 
    return distt
        
def select_centroids(data , k):
    index = np.random.choice(data.shape[0] , k , replace=False)
    centroids = data[index]
    return centroids

def make_clusters(data , centroids):
    # task is to assign each point of the data a cluster 
    cluster = []
    for point in data :
        distance = []
        for i in centroids:
            distance.append(euclid_distance(i , point))
        cluster.append(np.argmin(distance))
    return np.array(cluster)

def update_centroids(data , cluster , k):
    new_centroids = []
    
    for i in range(k):
        cluster_points = data[cluster == i]
        if len(cluster_points) > 0 : 
            new_centroids.append(np.mean(cluster_points , axis = 0))
        else : 
            new_centroids.append(data[np.random.choice(data.shape[0] , k , replace=False)])
    return np.array(new_centroids)
        
def KmeanS(data , k , max_iters=500 , tolerance = 1e-4):
    centroids = select_centroids(data, k)
    
    for i in range(max_iters):
        cluster = make_clusters(data, centroids)
        new_centroids = update_centroids(data , cluster , k)
        if(np.all(np.abs(centroids - new_centroids)) < tolerance):
            break 
        centroids = new_centroids
    return cluster , centroids 

def select_optimum_k(data):
    Max_K = 10 
    loss_list = []
    
    for i in range(1 , Max_K+1):
        cluster,centroids = KmeanS(data , i)
        inter_cluster_loss = 0 
        for i, centroid in enumerate(centroids): 
            points_in_cluster = data[cluster == i]
            for point in points_in_cluster:
                inter_cluster_loss += euclid_distance(point, centroid)
    
        loss_list.append(inter_cluster_loss)
    
    plt.plot(range(1 , Max_K+1) , loss_list, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inter-cluster Loss")
    plt.title("Elbow Method for Optimal k")
    plt.show()
    
    return loss_list
        
       
if __name__ == "__main__":
    #this is our data set 
    data = np.vstack([np.random.randn(100, 2) + [5, 5],
                  np.random.randn(100, 2) + [0, 0],
                  np.random.randn(100, 2) + [5, 0],
                  np.random.randn(100,2) + [-5,-5],
                  np.random.randn(100,2) + [10,10]])
    
    loss = select_optimum_k(data)
    optimumk = 3 #default value 
    
    for i in range(1,len(loss)):
        if(abs(loss[i] - loss[i-1]) < 100):
            optimumk = i 
            break 
        
    cluster , centroids = KmeanS(data , optimumk)
    
    #plot the results ; cluster contains numbers from 0 to k 
    
    for i in range(optimumk):
        cls_points = data[cluster == i]
        plt.scatter(cls_points[:,0] , cls_points[:,1] ,  label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
    plt.legend()
    plt.show()

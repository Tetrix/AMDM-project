import numpy as np
from numpy import linalg as LA
import pandas as pd
from sklearn.cluster import KMeans


# In[170]:


data_file = 'data/ca-HepTh.txt'



# In[171]:


data = pd.read_csv(data_file, sep = ' ', header = None)
left_nodes = data[0].values
right_nodes = data[1].values
combined_nodes = np.concatenate((left_nodes, right_nodes), axis = 0)
combined_nodes = np.unique(combined_nodes)


# In[172]:


def get_unique(arr):
    unique = []
    for i in arr:
        if i not in unique:
            unique.append(i)
    return unique


# In[173]:


def get_graph(combined_nodes, left_nodes, right_nodes):
    neighbors = []
    graph = {}

    for node in combined_nodes:
        if node in left_nodes:
            indices = np.where(left_nodes == node)
            for index in indices:
                neighbors.append(right_nodes[index])
        if node in right_nodes:
            indices = np.where(right_nodes == node)
            for index in indices:
                neighbors.append(left_nodes[index])

        unique_neighbors = get_unique(neighbors[0])

        unique_neighbors = np.array(unique_neighbors)
        graph[node] = unique_neighbors
        neighbors = []
    
    return graph


graph = get_graph(combined_nodes, left_nodes, right_nodes)


# In[174]:


def get_adjacency_matrix(graph):
    adjacency_matrix = np.zeros([len(graph), len(graph)])

    for key, value in graph.items():
        for neighbor in value:
            adjacency_matrix[key][value] = 1
    
    return np.array(adjacency_matrix)
    
adjacency_matrix = get_adjacency_matrix(graph)


# In[175]:


def get_degree_matrix(adjacency_matrix):
    degree_matrix = np.zeros([len(adjacency_matrix), len(adjacency_matrix)])
    
    for row_index in range(len(adjacency_matrix)):
        degree_matrix[row_index][row_index] = np.sum(adjacency_matrix[row_index])

    return degree_matrix

degree_matrix = get_degree_matrix(adjacency_matrix)




from scipy.sparse import csgraph

laplacian_matrix = csgraph.laplacian(adjacency_matrix, normed=True)


# In[ ]:


def get_eigen(laplacian_matrix):
    eig_values, eig_vectors = LA.eig(laplacian_matrix)
    return eig_values, eig_vectors

eig_values, eig_vectors = get_eigen(laplacian_matrix)


# In[162]:


def least_significat_eigens(k, eig_values, eig_vectors):
    idx = eig_values.argsort()[:k]
    eig_vectors = eig_vectors[:, idx]
    eig_values = eig_values[idx]
    
    return eig_vectors, eig_values


num_eigens = 20
least_vectors, least_values = least_significat_eigens(num_eigens, eig_values, eig_vectors)

least_values = least_values.real
least_vectors = least_vectors.real


# In[163]:


least_values



num_clusters = 20

kmeans = KMeans(
    n_clusters = num_clusters, 
    random_state = 0
)

kmeans.fit(least_vectors)



df = pd.DataFrame({'node' : combined_nodes, 'cluster' : kmeans.labels_})
df = df[['node', 'cluster']]
df.to_csv('output/ca-GrQc.output', index = False, sep = ' ')


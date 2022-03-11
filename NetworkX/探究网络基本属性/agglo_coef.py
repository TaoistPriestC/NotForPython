from platform import node
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 集聚系数(Agglomeration coefficient)
# 集聚系数也称集群系数,是用于描述一个图中顶点之间结集成团的程度系数
INF = int(1e9)


# 获取节点两两之间的最短路径
def floyd(weight_matrix):
    d = [[e if e != -1 else INF for e in row] for row in weight_matrix]
    nodes = len(weight_matrix)
    for k in range(nodes):
        for i in range(nodes):
            for j in range(nodes):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


# 计算网络的直径, 也即两个节点之间的最长路径
def get_diameter(shortest_path_matrix):
    maxv = 0
    for row in shortest_path_matrix:
        maxv = max(maxv, max(row))
    return maxv        

# 计算网络的平均距离
def get_avg_dist(shortest_path_matrix):
    sumv, nodes = 0, len(shortest_path_matrix)
    for row in shortest_path_matrix:
        sumv += sum(row)
    sumv /= 2 
    return 2 * sumv / nodes  / (nodes - 1)

if __name__ =="__main__":
    adjacency_matrix = [
         #1  #2  #3  #4  #5   #6
        [  0,  1,  1, -1,  1, -1], #1
        [  1,  0,  1,  1, -1, -1], #2
        [  1,  1,  0, -1, -1,  1], #3
        [ -1,  1, -1,  0, -1,  1], #4
        [  1, -1, -1, -1,  0, -1], #5
        [ -1, -1,  1,  1, -1,  0]  #6
    ]
    shortest_path_matrix = floyd(adjacency_matrix)
    print(np.array(shortest_path_matrix))
    print("Diameter = {}".format(get_diameter(shortest_path_matrix)))
    print("Average distance = {:.2f}".format(get_avg_dist(shortest_path_matrix)))
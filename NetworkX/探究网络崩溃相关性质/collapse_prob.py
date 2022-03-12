import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 自行构造一批数据集
# avgk = 1.5
# avgk = 2.50
# avgk = 3.33
# avgk = 3.75
# avgk = 4.00
# avgk = 5.00
# avgk = 5.875
# avgk = 7.00
# avgk = 10.15625

# neo_each_layer = [5, 2, 1]
# neo_each_layer = [3, 5, 2]
# neo_each_layer = [3, 5, 5, 2]
# neo_each_layer = [3, 5, 5, 5, 2]
# neo_each_layer = [3, 5, 5, 5, 5, 2]
# neo_each_layer = [3, 5, 5, 10, 5, 5, 5, 2]
# neo_each_layer = [3, 5, 5, 5, 10, 10, 2]
# neo_each_layer = [3, 10, 10, 5, 10, 10, 2]
# neo_each_layer = [3, 10, 12, 15, 12, 10, 2]


models_set = [
    [5, 2, 1],
    [3, 5, 2],
    [3, 5, 5, 2],
    [3, 5, 5, 5, 2],
    [3, 5, 5, 5, 5, 2],
    [3, 5, 5, 10, 5, 5, 5, 2],
    [3, 5, 5, 5, 10, 10, 2],
    [3, 10, 10, 5, 10, 10, 2],
    [3, 10, 12, 15, 12, 10, 2],
]


def construct_DGA(neo_each_layer):
    nodes = sum(neo_each_layer)
    adjacency_matrix  = [[0 for _ in range(nodes)] for _ in range(nodes)]
    beg, lt_end,  rt_end = 0, 0, 0
    for i,x in enumerate(neo_each_layer):
        if i + 1 < len(neo_each_layer):
            lt_end = beg + x
            rt_end = lt_end + neo_each_layer[i + 1]
            for j in range(beg, lt_end):
                for k in range(lt_end, rt_end):
                    adjacency_matrix[j][k] = 1
            beg = lt_end
    return np.array(adjacency_matrix)


def init_state(neo_each_layer, fai, miu, sigma):
    nodes = sum(neo_each_layer)
    node_threshold = np.zeros(nodes) + fai
    current_load = np.random.normal(miu, sigma, size = nodes)
    return node_threshold, current_load, nodes


def get_degree(adjacency_matrix):
    nodes = len(adjacency_matrix)
    indeg, outdeg, degree = list([0]*nodes), list([0]*nodes), 0
    for i in range(nodes):
        for j in range(nodes):
            if adjacency_matrix[i][j] == 1:
                outdeg[i] += 1
                indeg[j] += 1
                degree += 1
    return indeg, outdeg, degree/nodes



def bfs(adjacency_matrix, node_threshold, current_load, outdeg, src):
    """ Parameters description:
        :param adjacency_matrix: Adjaency matrix
        :param node_threshold: maintain threshold of each node
        :param current_load: current load of each node
        :param outdeg: outdeg of each node
        :param src: The source node of collapse
    """

    edges, damage = 0, 0
    vis, que = set(), list()
    
    vis.add(src)
    que.append(src)

    while len(que) > 0:
        size = len(que)
        for i in range(size):
            t = que.pop(0)
            if outdeg[t] != 0:
                tl = current_load[t] / outdeg[t]
                for i, e in enumerate(adjacency_matrix[t]):
                    if e == 1:
                        current_load[i] += tl
                        if current_load[i] > node_threshold[i] and i not in vis:
                            edges += 1
                            outdeg[t] -= 1
                            que.append(i)
                            vis.add(i)
            damage += 1
    # 返回受到破坏的边数与结点数,由于nodes可能引起歧义改用damage代表损坏的节点个数
    return edges, damage


# 按照节点的平均度区分颜色
def test_damage_prob(alpha, models_set):
    avgk_keys = list()
    data_memo = dict()
    p = lambda size, alpha : pow(size,-alpha)
    attack_force = np.arange(0.1, 1.5, 0.05)
    for neo_each_layer in models_set:
        adjacency_matrix = construct_DGA(neo_each_layer)
        indeg, outdeg, avgk = get_degree(adjacency_matrix)
        node_threshold, current_load, nodes = init_state(neo_each_layer, 0.4, 0.3, 0.05)
        src = int(np.random.choice(np.arange(neo_each_layer[0]),1))
        
        avgk_keys.append(str("{:.3f}".format(avgk)))
        print("Now <k> = {key}".format(key = avgk_keys[-1]))

        s_list, ps_list = [], [] 
        for af in attack_force:
            print(" --> 攻击强度:{:.3f}".format(af))        
            current_load[src] = af

            ## Get the number of corrupted edges and nodes
            e,s = bfs(adjacency_matrix[:], node_threshold[:], current_load[:], outdeg, src)

            # 如果s代表崩溃的连通分量大小
            ps = p(s, 1.5)
            s_list.append(s)
            ps_list.append(ps)

            # 如果s代表剩下的连通分量大小

        data_memo[avgk_keys[-1]] = (s_list, ps_list)
    for keys, sps in data_memo.items():
        s, ps = sps
        print("Now <k> = {}".format(keys))
        print(s)
        print(ps)
        
        plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
        
        plt.xlabel(r"$S$")
        plt.ylabel(r"$P(S) \sim S^{-\alpha}$")
        
        # plt.xlabel(r"崩溃规模 $S$")
        # plt.ylabel(r"崩溃规模的发生概率 $P(S)$")
        plt.scatter(s, ps, label = str(r"$\langle k \rangle$ = " + keys))
    plt.legend([str(r"$\langle k \rangle$ = " + keys)  for keys in data_memo.keys()])
    plt.savefig("崩溃规模概率.png")
    plt.show()


if __name__ == "__main__":
    # for neo_each_layer in models_set:
    #     adjacency_matrix = construct_DGA(neo_each_layer)
    #     indeg, outdeg, avgk = get_degree(adjacency_matrix)
    #     print(avgk)
    np.random.seed(0)
    test_damage_prob(1.5, models_set[:])
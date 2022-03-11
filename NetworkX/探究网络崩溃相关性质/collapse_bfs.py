import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -- Constant variable beg --- 
EPS = 1e-5
FAI = 0.40
NEXP = 20
RT_XLOC = 0.65
RT_YLOC = 0.90
FONT_DICT  = dict(fontsize = 8, color = "k", family="Microsoft YaHei")
BGBOX = {
    "facecolor": "#7B68EE", # 填充色
    'edgecolor':'b',        # 外框色
    'alpha': 0.5,           # 框透明度
    'pad': 8,               # 本文与框周围距离 
}
# --- Constant variable end ---



def get_degree(adjacency_matrix, node_num):
    indeg = [0] * node_num 
    outdeg = [0] * node_num
    edges, avgK = 0, 0
    for i in range(node_num):
        for j in range(node_num):
            indeg[j] += adjacency_matrix[i][j] 
            outdeg[i] += adjacency_matrix[i][j] 
            edges += adjacency_matrix[i][j]
    avgK = 2 * edges / node_num
    # Maybe average degree would be used in some scene!
    return indeg, outdeg


def bfs(adjacency_matrix, node_threshold, current_load, node_num, src):
    """ Parameters description:
        :param adjacency_matrix: Adjaency matrix
        :param node_threshold: maintain threshold of each node
        :param src: The source node of collapse
    """

    intact = node_num
    layers, edges, avgK = 0, 0, 0
    vis, que = set(),list()
    indeg, outdeg = get_degree(adjacency_matrix, node_num)
    
    vis.add(src)
    que.append(src)

    while len(que) > 0:
        ## Get the front element from the queue and distribute the load to adjacent nodes
        layers += 1
        size = len(que)
        print("The {:0>2d} layers: ".format(layers), end = "\0")
        for i in range(size):
            ## Print the collapsed node
            t = que.pop(0)
            print("{:0>2d} ".format(t), end = " ")

            ## Confirm that zero is not allow for out degree, 
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
                        ## edges and outdeg[t] are only modified when the threshold was exceeded!
            intact -= 1            
            ## then check the adjacency edge if it's connected (e==1), 
            ## those connected node need to share the pressure of collapsed nodes!
        print()
        print("Now the current load of each nodes :", current_load)

    if intact > 0:
        print("The network backed to be stable!")
    else:
        print("The network has been destroyed!")
    avgK =  edges / (node_num - intact)
    return avgK


def test_avgK_FAI(node_num, lo_fai, hi_fai, step_fai, miu):
    ## To record the result
    _fai_List = list()
    _avgK_List = list()

    np.random.seed(0)

    FaiList = np.arange(lo_fai, hi_fai, step_fai)
    for idx,fai in enumerate(FaiList):
        ## Modify the threshold and keep other variables unchanged
        ## Setting a FAI list
        adjacency_matrix = np.random.randint(0, 2, size = (node_num * node_num)).reshape(node_num, node_num)
        current_load = np.random.normal(miu, 0.1, size = node_num)        
        node_threshold = np.zeros(node_num) + fai
        
       
        ## Randomly choose and attack one node
        src = np.random.choice(np.arange(node_num), 1)[0]
        current_load[src] = 1.0
        
        print()
        print("+-------------------------------------- Simulation:{:3d} ---------------------------------------+".format(idx))
        avgK = bfs(adjacency_matrix, node_threshold, current_load, node_num, src)
        _fai_List.append(fai)
        _avgK_List.append(avgK)
        print("+---------------------------------------------------------------------------------------------+")
        print()
    return np.array(_fai_List), np.array(_avgK_List)
    


def test_with_diff_node(lo_fai, hi_fai):
    plt.ion()
    node_num_list = [10, 20, 30, 50, 100, 150, 200]
    miu_list = [0.1, 0.15, 0.2, 0.25, 0.30]
    step_list = [0.1, 0.05, 0.01]
    count, node_group, step_group, miu_group = 0, 0, 0, 0
    for node_num in node_num_list:
        node_group, step_group = node_group + 1, 0
        for step in step_list:
            step_group, miu_group = step_group + 1, 0
            for miu in miu_list:
                miu_group += 1
                count += 1
                print("第{:0>4d}组 - 结点个数={:0>3d} - 正态分布均值={:.2f} - 步长={:.2f}".format(count, node_num, miu, step))
                fai_array, avgK_array = test_avgK_FAI(node_num, lo_fai, hi_fai, step, miu)     
                print(list(fai_array))
                print(list(avgK_array)) 
                plt.clf()
                plt.xlabel(r"$Threshold - \varphi$")
                plt.ylabel(r"$Averagedegree-of-collapse-part \langle k \rangle $")
                plt.text(RT_XLOC, RT_YLOC, "Network Size(Nodes):{:3d}".format(node_num)
                   , transform = plt.gca().transAxes
                   , fontdict = FONT_DICT
                   , bbox = BGBOX
                )
                plt.plot(fai_array,avgK_array)
                plt.savefig("./第{:0>4d}组-n{:0>2d}-s{:0>2d}-m{:0>2d}.png".format(count, node_group, step_group, miu_group))
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    """
    node_num = 20

    records = int(((hi_fai - lo_fai) / step_fai))

    _fai_array = np.zeros(records)
    _avgK_array = np.zeros(records)
    
    
    _fai_array_t, _avgK_array_t = test_avgK_FAI(node_num, lo_fai, hi_fai, step_fai, 0.3) 
    
    _fai_array = _fai_array_t
    _avgK_array = _avgK_array_t

    print(list(_fai_array))
    print(list(_avgK_array))
    """
    test_with_diff_node(0.1, 1.0)
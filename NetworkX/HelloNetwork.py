from turtle import pos
import networkx as nx
import matplotlib.pyplot as plt

G = nx.cubical_graph()
subax1 = plt.subplot(121)
nx.draw(G)
subax2 = plt.subplot(122)
nx.draw(G, pos=nx.circular_layout(G), node_color = 'r', edge_color = 'b')
plt.show()
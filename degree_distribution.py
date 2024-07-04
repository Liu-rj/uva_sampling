import torch
import dgl
import numpy as np
import os
import matplotlib.pyplot as plt


def load_graph_csaw(beg_file, adj_file):
    beg_pos = np.fromfile(beg_file, dtype=np.int64)
    adj_list = np.fromfile(adj_file, dtype=np.int64)
    g = dgl.graph(("csr", (beg_pos, adj_list, [])))
    return g


sorted_degree_list = []
x_list = []
dir_list = [
    # "/home/ubuntu/C-SAW/streaming/dataset/web-google",
    # "/home/ubuntu/C-SAW/streaming/dataset/livejournal",
    # "/home/ubuntu/C-SAW/streaming/dataset/reddit",
    # "/home/ubuntu/C-SAW/streaming/dataset/ogbn_products",
    "/home/ubuntu/C-SAW/streaming/dataset/friendster",
]
for dir in dir_list:
    print("dataset: ", dir)
    g: dgl.DGLGraph = load_graph_csaw(
        os.path.join(dir, "beg_pos.bin"), os.path.join(dir, "csr.bin")
    ).long()
    x_list.append(torch.arange(0, g.num_nodes()) / g.num_nodes())
    sorted_degree = torch.sort(g.out_degrees(), descending=True)
    sorted_degree_list.append(sorted_degree.values.numpy() / g.num_edges())

# plot the degree distribution
plt.figure()
for i, x in enumerate(x_list):
    plt.plot(x, sorted_degree_list[i], label=dir_list[i].split("/")[-1])
plt.legend(loc="upper right")
plt.grid()
plt.xlabel("CDF")
plt.ylabel("Degree")
plt.title("Degree Distribution")
plt.savefig("figs/degree_distribution.pdf")

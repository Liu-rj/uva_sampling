import os
import torch
import dgl
import pandas


def load_graph_txt(txt_file, skiprows=0, undirected=False):
    df = pandas.read_csv(
        txt_file,
        sep="\t",
        skiprows=skiprows,
        header=None,
        names=["src", "dst"],
    )
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    g = dgl.graph((src, dst))
    del df
    print("compact the graph")
    g = dgl.compact_graphs(g)
    if undirected:
        g = dgl.to_bidirected(g)
    print(g)
    return g


def save_graph_topo(g, dataset):
    dir = f"/home/ubuntu/C-SAW/streaming/dataset/{dataset}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    beg_file = os.path.join(dir, "beg_pos.bin")
    adj_file = os.path.join(dir, "csr.bin")
    row_ptr, indices, _ = g.adj_tensors("csr")
    row_ptr.to(torch.int64).numpy().tofile(beg_file)
    indices.to(torch.int64).numpy().tofile(adj_file)


# data = RedditDataset()
# g = data[0]
# print("Dataset: Reddit")

# data = DglNodePropPredDataset(name="ogbn-products", root="/home/ubuntu/dataset/")
# g, labels = data[0]
# print("Dataset: ogbn-products")

# txt_file = "web-google/web-Google.txt"
# txt_file = "livejournal/soc-LiveJournal1.txt"
txt_file = "friendster/com-friendster.ungraph.txt"
print(os.path.dirname(txt_file))

g = load_graph_txt(
    os.path.join("/home/ubuntu/C-SAW/streaming/dataset", txt_file),
    skiprows=4,
    undirected="friendster" in txt_file,
)
save_graph_topo(g, os.path.dirname(txt_file))

import torch
import dgl
from dgl.sampling import random_walk
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import os
import numpy as np
from tqdm import tqdm
import time
import csv


def deterministic_walk(g: dgl.DGLGraph, seeds, length):
    row_ptr, indices, _ = g.adj_tensors("csr")
    trace = torch.full(
        (seeds.numel(), length + 1), -1, dtype=torch.int64, device=seeds.device
    )
    trace[:, 0] = seeds
    for i in tqdm(range(1, length + 1), ncols=100):
        for j in range(seeds.numel()):
            if trace[j, i - 1] == -1:
                continue
            u = trace[j, i - 1]
            neighbor_len = row_ptr[u + 1] - row_ptr[u]
            if neighbor_len > 0:
                neighbors = indices[row_ptr[u] : row_ptr[u + 1]]
                trace[j, i] = neighbors[0]
    return trace, _


def load_graph_csaw(beg_file, adj_file):
    beg_pos = np.fromfile(beg_file, dtype=np.int64)
    adj_list = np.fromfile(adj_file, dtype=np.int64)
    g = dgl.graph(("csr", (beg_pos, adj_list, [])))
    print(g)
    return g


dir_list = [
    # "/home/ubuntu/C-SAW/streaming/dataset/web-google",
    "/home/ubuntu/C-SAW/streaming/dataset/livejournal",
    # "/home/ubuntu/C-SAW/streaming/dataset/reddit",
    # "/home/ubuntu/C-SAW/streaming/dataset/ogbn_products",
    # "/home/ubuntu/C-SAW/streaming/dataset/friendster",
]

for sampling_type in [
    "GPU",
    "UVA",
]:
    for dir in dir_list:
        print("dataset: ", dir)
        g: dgl.DGLGraph = load_graph_csaw(
            os.path.join(dir, "beg_pos.bin"), os.path.join(dir, "csr.bin")
        ).long()

        if sampling_type == "GPU":
            g = g.to("cuda")
        elif sampling_type == "UVA":
            g.pin_memory_()
        else:
            raise ValueError("Invalid sampling type")

        # g.ndata["degree"] = g.out_degrees().float()
        # g.apply_edges(lambda edges: {"weight": edges.dst["degree"]})

        print(
            f"Nodes: {g.num_nodes()}, Edges: {g.num_edges()}, Degree: {g.num_edges() / g.num_nodes()}"
        )
        settings = [
            (4000, 2000),
            (10000, 100),
            (10000, 2000),
        ]
        for setting in settings:
            time_list = []
            for i in range(10):
                # seeds = np.fromfile(os.path.join(dir, f"seeds_{setting[0]}.bin"), dtype=np.int64)
                # seeds = torch.from_numpy(seeds).to("cuda")
                seeds = torch.randint(
                    0, g.num_nodes(), (setting[0],), dtype=torch.int64, device="cuda"
                )
                # seeds_numpy = seeds.cpu().numpy()
                # seeds_numpy.tofile(os.path.join(dir, f"seeds_{setting[0]}.bin"))

                torch.cuda.synchronize()
                start = time.time()
                ret = dgl.sampling.random_walk(g, seeds, length=setting[1], prob=None)
                # ret = deterministic_walk(g, seeds, setting[1])
                torch.cuda.synchronize()
                end = time.time()
                time_list.append(end - start)
                # print(f"Time: {end - start:.4f}s")
            avg_time = np.mean(time_list[1:])
            trace = ret[0]
            touched_frontiers = torch.unique(trace[:, :-1])
            num_sampled_edge = torch.sum(trace != -1).item() - setting[0]

            log_data = [
                sampling_type,
                setting[0],
                setting[1],
                touched_frontiers.numel(),
                touched_frontiers.numel() / g.num_nodes(),
                num_sampled_edge,
                round(avg_time, 4),
            ]
            print(log_data)

            # write to csv log file
            with open("logs/randomwalk.csv", "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                log_info = [
                    os.path.basename(dir),
                    g.num_nodes(),
                    g.num_edges(),
                ] + log_data
                writer.writerow(log_info)

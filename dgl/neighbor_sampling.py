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
        g = load_graph_csaw(
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
            (1024, [10, 10]),
            (1024, [10, 10, 10]),
            (4096, [10, 10, 10]),
        ]
        for setting in settings:
            time_list = []
            for i in range(10):
                seeds = torch.randint(
                    0, g.num_nodes(), (setting[0],), dtype=torch.int64, device="cuda"
                )

                torch.cuda.synchronize()
                start = time.time()
                blocks = []
                for fanout in setting[1]:
                    sg: dgl.DGLGraph = dgl.sampling.sample_neighbors(
                        g, seeds, fanout=fanout, edge_dir="out", prob=None, replace=True
                    )
                    seeds = sg.edges()[1]
                    blocks.append(sg)
                torch.cuda.synchronize()
                end = time.time()
                time_list.append(end - start)
                # print(f"Time: {end - start:.4f}s")
            avg_time = np.mean(time_list[1:])
            frontier_counts = torch.zeros(
                g.num_nodes(), dtype=torch.int64, device="cuda"
            )
            num_sampled_edge = 0
            for sg in blocks:
                src, dst = sg.edges()
                frontier_counts[src] += 1
                num_sampled_edge += sg.num_edges()
            num_touched_frontier = (frontier_counts > 0).sum().item()

            log_data = [
                sampling_type,
                setting[0],
                setting[1],
                num_touched_frontier,
                num_touched_frontier / g.num_nodes(),
                num_sampled_edge,
                round(avg_time, 4),
            ]
            print(log_data)

            # write to csv log file
            with open("logs/neighbor_sampling.csv", "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                log_info = [
                    os.path.basename(dir),
                    g.num_nodes(),
                    g.num_edges(),
                ] + log_data
                writer.writerow(log_info)

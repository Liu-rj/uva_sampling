import gs
import torch
from gs.utils import load_graph
from typing import List, Union
import numpy as np
import dgl
import os
import time
from gs.format import _CSC, _COO
import csv


def load_graph_csaw(beg_file, adj_file):
    beg_pos = np.fromfile(beg_file, dtype=np.int64)
    adj_list = np.fromfile(adj_file, dtype=np.int64)
    g = dgl.graph(("csr", (beg_pos, adj_list, [])))
    print(g)
    return g


def randomwalk_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor, walk_length: int):
    paths = A.random_walk(seeds, walk_length)
    return paths


dir_list = [
    "/home/ubuntu/C-SAW/streaming/dataset/web-google",
    "/home/ubuntu/C-SAW/streaming/dataset/livejournal",
    "/home/ubuntu/C-SAW/streaming/dataset/reddit",
    "/home/ubuntu/C-SAW/streaming/dataset/ogbn_products",
    "/home/ubuntu/C-SAW/streaming/dataset/friendster",
]

for sampling_type in [
    "GPU",
    "UVA",
]:
    for dir in dir_list:
        if "friendster" in dir and sampling_type == "GPU":
            continue
        print("dataset: ", dir)
        g = load_graph_csaw(
            os.path.join(dir, "beg_pos.bin"), os.path.join(dir, "csr.bin")
        ).long()
        print(
            f"Nodes: {g.num_nodes()}, Edges: {g.num_edges()}, Degree: {g.num_edges() / g.num_nodes()}"
        )
        csr_indptr, csr_indices, _ = g.adj_tensors("csr")

        m = gs.matrix_api.Matrix()
        if sampling_type == "GPU":
            m.load_graph("CSC", [csr_indptr.cuda(), csr_indices.cuda()])
        elif sampling_type == "UVA":
            m.load_graph("CSC", [csr_indptr.pin_memory(), csr_indices.pin_memory()])
        else:
            raise ValueError("Invalid sampling type")

        settings = [
            (4000, 2000),
            (10000, 100),
            (10000, 2000),
        ]
        for setting in settings:
            time_list = []
            for i in range(10):
                seeds = torch.randint(
                    0, g.num_nodes(), (setting[0],), dtype=torch.int64, device="cuda"
                )

                torch.cuda.synchronize()
                start = time.time()
                paths = randomwalk_sampler(m, seeds, setting[1])
                torch.cuda.synchronize()
                end = time.time()
                time_list.append(end - start)

            avg_time = np.mean(time_list[1:])
            frontier_counts = torch.zeros(
                g.num_nodes(), dtype=torch.int64, device="cuda"
            )
            num_touched_frontier = torch.unique(paths[:-1, :]).numel()
            num_sampled_edge = torch.sum(paths != -1).item() - setting[0]

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
            with open("logs/randomwalk.csv", "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                log_info = [
                    os.path.basename(dir),
                    g.num_nodes(),
                    g.num_edges(),
                ] + log_data
                writer.writerow(log_info)

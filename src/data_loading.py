"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import os

from torch_geometric.datasets import Planetoid, MixHopSyntheticDataset
import torch_geometric.transforms as transforms
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from data_utils import keep_only_largest_connected_component

DATA_PATH = "data"


def get_dataset(name: str, use_lcc: bool = True, homophily=None, use_gdc: bool = True):
    path = os.path.join(DATA_PATH, name)
    evaluator = None

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(path, name)
    elif name in ["OGBN-Arxiv", "OGBN-Products"]:
        dataset = PygNodePropPredDataset(name=name.lower(), transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name=name.lower())
        use_lcc = False
    elif name == "MixHopSynthetic":
        dataset = MixHopSyntheticDataset(path, homophily=homophily)
    else:
        raise Exception("Unknown dataset.")

    if use_lcc:
        dataset = keep_only_largest_connected_component(dataset)

    if use_gdc:
        print("I am using gdc")
        diffusiontransform = transforms.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        dataset.data = diffusiontransform(dataset.data)

    # Make graph undirected so that we have edges for both directions and add self loops
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])

    

    return dataset, evaluator
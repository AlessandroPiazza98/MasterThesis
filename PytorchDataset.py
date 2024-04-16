from torch_skeleton.datasets import NTU, BABEL
import torch_skeleton.transforms as T
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pickle
import numpy as np
import argparse
from torch.utils.data import Dataset


class BabelConvToPyG:
    """Export the data as PyTorch Geometric Data
    The 1. Configuration of 25 body joints in our dataset. The labels of the joints are: 1-base of the spine 2-middle of the spine
    3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8-
    left hand 9-right shoulder 10-right elbow 11-right wrist 12-
    right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17-
    right hip 18-right knee 19-right ankle 20-right foot 21-spine 22-
    tip of the left hand 23-left thumb 24-tip of the right hand 25-
    right thumb """

    def __call__(self, x):
        kpts = 25
        frames = 150

        # Spatial edges according to kpts
        # sp_start_i=[1,1,1,2,2,3,3,4,5,5,7,7,8,8,8,9,9,10,10,11,11,12,12,12,13,13,14,14,15,15,16,17,17,18,18,19,19,20,21,21,21,21,22,23,24,25]
        sp_start_i = [1, 2, 3, 3, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 17, 18, 19]
        # Recounting to make edge lists start from 0
        sp_start_i = [x - 1 for x in sp_start_i]

        sp_edge_i = []

        # Create the edges for all the frames
        for i in range(frames):
            sp_edge_i = sp_edge_i + [j + 25 * i for j in sp_start_i]

        # sp_start_o=[2,13,17,1,21,4,21,3,6,21,6,8,7,22,23,10,21,9,11,10,12,11,24,25,1,14,13,15,14,16,15,1,18,17,19,18,20,19,2,3,5,9,8,8,12,12]
        sp_start_o = [2, 21, 4, 21, 6, 21, 7, 8, 22, 23, 21, 10, 11, 12, 24, 25, 14, 15, 16, 18, 19, 20]

        sp_start_o = [x - 1 for x in sp_start_o]

        sp_edge_o = []
        for i in range(frames):
            sp_edge_o = sp_edge_o + [j + 25 * i for j in sp_start_o]

        # Temporal edges
        tmp_start_i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        tmp_start_i = [x - 1 for x in tmp_start_i]

        # Create the edges between all the frames
        tmp_edge_i = []
        for i in range(frames - 1):
            tmp_edge_i = tmp_edge_i + [j + 25 * i for j in tmp_start_i]

        tmp_start_o = [j + 25 for j in tmp_start_i]

        tmp_edge_o = []
        for i in range(frames - 1):
            tmp_edge_o = tmp_edge_o + [j + 25 * i for j in tmp_start_o]

        # Put edge lists in edge_index tensor
        edge_index = torch.tensor([sp_edge_i + tmp_edge_i,
                                   sp_edge_o + tmp_edge_o])

        # Create edge_attr: 0 for spatial edges, 1 for temporal edges
        edge_attr = torch.tensor([[0.]] * len(sp_edge_i) + [[1.]] * len(tmp_edge_i))

        edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)

        nodes = []

        # Extract the features from x and put on a tensor: for Babel from (1,150,25,3) to (3750,3)
        x = x[0]
        for i in range(frames):
            for j in range(kpts):
                nodes.append(x[i][j].tolist())

        x = torch.tensor(nodes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


class NTUConvToPyG:
    """Export the data as PyTorch Geometric Data
    The 1. Configuration of 25 body joints in our dataset. The labels of the joints are: 1-base of the spine 2-middle of the spine
    3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8-
    left hand 9-right shoulder 10-right elbow 11-right wrist 12-
    right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17-
    right hip 18-right knee 19-right ankle 20-right foot 21-spine 22-
    tip of the left hand 23-left thumb 24-tip of the right hand 25-
    right thumb """

    def __call__(self, x):
        kpts = 25
        frames = x.shape[1]

        # Spatial edges according to kpts
        # sp_start_i=[1,1,1,2,2,3,3,4,5,5,7,7,8,8,8,9,9,10,10,11,11,12,12,12,13,13,14,14,15,15,16,17,17,18,18,19,19,20,21,21,21,21,22,23,24,25]
        sp_start_i = [1, 2, 3, 3, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 17, 18, 19]
        # Recounting to make edge lists start from 0
        sp_start_i = [x - 1 for x in sp_start_i]

        sp_edge_i = []

        # Create the edges for all the frames
        for i in range(frames):
            sp_edge_i = sp_edge_i + [j + 25 * i for j in sp_start_i]

        # sp_start_o=[2,13,17,1,21,4,21,3,6,21,6,8,7,22,23,10,21,9,11,10,12,11,24,25,1,14,13,15,14,16,15,1,18,17,19,18,20,19,2,3,5,9,8,8,12,12]
        sp_start_o = [2, 21, 4, 21, 6, 21, 7, 8, 22, 23, 21, 10, 11, 12, 24, 25, 14, 15, 16, 18, 19, 20]

        sp_start_o = [x - 1 for x in sp_start_o]

        sp_edge_o = []
        for i in range(frames):
            sp_edge_o = sp_edge_o + [j + 25 * i for j in sp_start_o]

        # Temporal edges
        tmp_start_i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        tmp_start_i = [x - 1 for x in tmp_start_i]

        # Create the edges between all the frames
        tmp_edge_i = []
        for i in range(frames - 1):
            tmp_edge_i = tmp_edge_i + [j + 25 * i for j in tmp_start_i]

        tmp_start_o = [j + 25 for j in tmp_start_i]

        tmp_edge_o = []
        for i in range(frames - 1):
            tmp_edge_o = tmp_edge_o + [j + 25 * i for j in tmp_start_o]

        # Put edge lists in edge_index tensor
        edge_index = torch.tensor([sp_edge_i + tmp_edge_i,
                                   sp_edge_o + tmp_edge_o])

        # Create edge_attr: 0 for spatial edges, 1 for temporal edges
        edge_attr = torch.tensor([[0.]] * len(sp_edge_i) + [[1.]] * len(tmp_edge_i))

        edge_index, edge_attr = to_undirected(edge_index=edge_index, edge_attr=edge_attr)

        nodes = []

        # Extract the features from x and put on a tensor: for Babel from (1,150,25,3) to (3750,3)
        x = x[0]
        for i in range(frames):
            for j in range(kpts):
                nodes.append(x[i][j].tolist())

        x = torch.tensor(nodes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data



class SkeletalDataset(Dataset):
    """NTU or Babel dataset."""

    def __init__(self, name, data_size, classes, split, eval_type, root_dir):

        self.data_size = data_size
        if name == "Babel":

            # download babel skeleton dataset
            babel = BABEL(
                root=root_dir,
                num_classes=int(classes),
                split=split,
                transform=T.Compose([
                    T.Denoise(),
                    T.CenterJoint(),
                    T.SplitFrames(),
                    BabelConvToPyG()
                ]),
            )

            self.dataset = babel

        elif name == "NTU":
            # download babel skeleton dataset
            ntu = NTU(
                root=root_dir,
                num_classes=int(classes),
                eval_type=eval_type,
                split=split,
                transform=T.Compose([
                    T.Denoise(),
                    T.CenterJoint(),
                    T.SelectKBodies(1),
                    NTUConvToPyG()
                ]),
            )


            self.dataset = ntu

    def __len__(self):

        if self.data_size=="Small":
            return 200

        elif self.data_size=="Medium":
            return 2000

        return len(self.dataset)


    def __getitem__(self, idx):
        data, y =self.dataset[idx]
        data.y = torch.tensor([y])

        return data




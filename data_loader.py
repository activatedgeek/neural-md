import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PDBChainsDataLoader(Dataset):
    def __init__(self, cluster_path, chains_path, cluster_range):
        """

        :param cluster_path: path to cluster file
        :param chains_path: path to chains folder
        :param cluster_range: list of cluster id's to load via the data loader
        """
        self.cluster_path = cluster_path
        self.chains_path = chains_path

        # Get a list of all chains in cluster_range, e.g. all chains in clusters [1,5,7]
        self.chains = []
        with open(cluster_path, 'r') as f:
            clust_id = 0
            for clust in f:
                if clust_id in cluster_range:
                    self.chains.extend(clust.split())
                clust_id += 1

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, item):
        """
        @NOTE: Only reading chain file here to be memory efficient
        """
        chain_id = self.chains[item]
        pdb_id = chain_id.split('_')
        npy_path = os.path.join(self.chains_path, pdb_id, chain_id, '.npy')
        chain_data = np.load(npy_path)
        return torch.from_numpy(chain_data)

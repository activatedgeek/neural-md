import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PDBChainsDataLoader(Dataset):
    def __init__(self, cluster_path, chains_path, cluster_ids):
        """

        :param cluster_path: path to cluster file, one cluster per line, space separated chain ids
        :param chains_path: path to chains folder
        :param cluster_ids: list of cluster id's to load via the data loader
        """
        self.cluster_path = cluster_path
        self.chains_path = chains_path

        # Get a list of all chains in cluster_range, e.g. all chains in clusters [1,5,7]
        self.chains = []
        with open(cluster_path, 'r') as f:
            clust_id = 0
            for clust in f:
                if clust_id in cluster_ids:
                    self.chains.extend(clust.split())
                clust_id += 1

    def __len__(self):
        return len(self.chains)

    def get_seq_from_fasta(self, path, chain):
        seq_string = None
        with open(path, 'r') as f:
            seq_id = None
            for line in f:
                if line[0] == '>':
                    seq_id = line[1:].split('|')[0].replace(':', '_')
                else:
                    if seq_id == chain:
                        seq_string = line

        assert seq_string is not None, 'Could not get sequence for chain {}'.format(chain)
        return seq_string

    def __getitem__(self, item):
        """
        @NOTE: Only reading chain file here to be memory efficient
        """
        chain = self.chains[item]
        pdb_id, chain_id = chain.split('_')
        npy_path = os.path.join(self.chains_path, pdb_id, chain + '.npy')
        chain_data = np.load(npy_path)
        chain_tensor = torch.from_numpy(chain_data)
        fasta_path = os.path.join(self.chains_path, pdb_id, pdb_id + '.fst')
        seq_string = self.get_seq_from_fasta(fasta_path, chain)
        return chain_tensor, seq_string

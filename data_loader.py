import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PDBChainsDataLoader(Dataset):
    def __init__(self, chains_path):
        """
        Takes a folder which contains chain ".npy" files and ".fst" files
        :param chains_path: path to chains folder
        """
        self.chains_path = chains_path

        # Get a list of all chains
        self.chains = list(
            map(
                lambda f: os.path.splitext(f)[0],
                filter(
                    lambda f: '.fst' in f,
                    os.listdir(chains_path)
                )
            )
        )

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, item):
        """
        @NOTE: Only reading chain file here to be memory efficient
        """
        chain = self.chains[item]
        npy_path = os.path.join(self.chains_path, chain + '.npy')
        chain_data = np.load(npy_path)
        chain_tensor = torch.from_numpy(chain_data)
        fasta_path = os.path.join(self.chains_path, chain + '.fst')
        with open(fasta_path, 'r') as f:
            seq_string = f.readlines()[1].rstrip('\n')
        return chain_tensor, seq_string

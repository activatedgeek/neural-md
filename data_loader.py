import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PDBChainsDataLoader(Dataset):
    NPY_EXT = '.npy'
    FASTA_EXT = '.faa'

    def __init__(self, chains_path):
        """
        Takes a folder which contains chain ".npy" files and ".fst" files
        :param chains_path: path to chains folder
        """
        self.chains_path = chains_path

        # Get a list of all chains from all FASTA files
        self.chains = []
        # self.seq_strings = []

        for f_name in os.listdir(self.chains_path):
            if self.FASTA_EXT in f_name:
                faa_path = os.path.join(self.chains_path, f_name)
                with open(faa_path, 'r') as fasta:
                    for line in fasta:
                        if line[0] == '>':
                            chain_id = line[1:].split('|')[0]
                            self.chains.append(chain_id)
                        # else:
                        #     self.seq_strings.append(line)

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, item):
        """
        @NOTE: Only reading numpy file here to be memory efficient
        """
        chain_id = self.chains[item]
        npy_path = os.path.join(self.chains_path, chain_id + self.NPY_EXT)
        chain_data = np.load(npy_path)
        chain_tensor = torch.from_numpy(chain_data).float()

        # seq_string = self.seq_strings[item]

        return chain_tensor

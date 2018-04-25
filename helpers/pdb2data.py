import os
import math
import numpy as np
import threading
import pyrosetta
from neural_md import make_fasta, pdb_to_chains

NUM_THREADS = 5

PDB_DIR = 'data/pdb'
CHAINS_DIR = 'data/chains'


def process_pdb_list(files):
    try:
        os.makedirs(CHAINS_DIR)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise

    processed_count = 0
    for f in files:
        pdb_id = os.path.splitext(f)[0]
        pdb_path = os.path.join(PDB_DIR, f)

        out_dir = os.path.join(CHAINS_DIR, pdb_id)
        if os.path.isdir(out_dir):
            print('{} already processed'.format(pdb_path))
            continue
        else:
            os.makedirs(out_dir)

        faa_path = os.path.join(out_dir, pdb_id + '.faa')

        with open(faa_path, 'w') as f:
            for chain_id, seq_string, chain_list in pdb_to_chains(pdb_path):
                fasta_string = make_fasta(pdb_id, chain_id, seq_string)
                f.write(fasta_string)

                npy_path = os.path.join(out_dir, '{}_{}.npy'.format(pdb_id, chain_id))
                np.save(npy_path, np.array(chain_list).T) # 109 x 64 numpy array

        print('Processed {}'.format(pdb_path))
        processed_count += 1

    print('{}/{} new PDBs processed'.format(processed_count, len(files)))


class PDB2DataThread(threading.Thread):
    def __init__(self, pdb_files):
        super(PDB2DataThread, self).__init__()

        self.pdb_files = pdb_files

    def run(self):
        process_pdb_list(self.pdb_files)


def main():
    # Needed for PyRosetta to work
    pyrosetta.init()

    pdb_files = list(filter(lambda f: '.pdb' in f, os.listdir(PDB_DIR)))
    total = len(pdb_files)
    batch = math.ceil(total / NUM_THREADS)

    thread_list = []
    for i in range(NUM_THREADS):
        pdb_file_batch = pdb_files[i * batch: (i + 1) * batch]
        thread_list.append(PDB2DataThread(pdb_file_batch))

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()


if __name__ == '__main__':
    main()

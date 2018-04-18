import os
import math
import numpy as np
import threading
import pyrosetta

NUM_THREADS = 1

PDB_DIR = 'data/pdb'
CHAINS_DIR = 'data/chains'

# maximum count for atoms and chi angles (see get_max_counts.py)
MAX_ATOM_COUNT = 27
MAX_CHI_COUNT = 4

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W',
    'X' # Unknown
]


def aa_to_vector(pose, r_i):
    """
    Convert Amino Acids to vector of angles, atom coordinates and one-hot encoded letter type
    :param pose:
    :param r_i:
    :return:
    """
    residue = pose.residue(r_i)

    # residue-type
    letter = residue.name1()
    assert letter in RESIDUE_LETTERS, '{} residue not supported'.format(residue.annotated_name())

    # angle parameters
    angles = [
        pose.phi(r_i),
        pose.psi(r_i),
        pose.omega(r_i),
    ]

    chi_angles = [
        *residue.chi(),
        # append extra zeros
        *([0.0] * (MAX_CHI_COUNT - len(residue.chi())))
    ]

    # normalize angles (-PI to PI)
    angles = list(map(lambda a: a / 180.0, angles))

    # atom coordinates, concatenation of x,y,z of each atom
    coords = []
    for atom in residue.atoms():
        coords.extend(atom.xyz())

    # append extra zeros for size consistency
    coords.extend([0.0, 0.0, 0.0] * (MAX_ATOM_COUNT - residue.natoms()))

    one_hot_type = [0] * len(RESIDUE_LETTERS)
    one_hot_type[RESIDUE_LETTERS.index(letter)] = 1

    feature_vector = [*coords, *angles, *chi_angles, *one_hot_type]
    return letter, feature_vector


def pdb_to_chains(pdb_path):
    pdb_file = os.path.basename(pdb_path)
    pdb_id, _ = os.path.splitext(pdb_file)

    # Generate chains from PDB
    pose = pyrosetta.pose_from_pdb(pdb_path)
    chains = {}
    seqs = {}
    for i in range(1, pose.total_residue() + 1):
        try:
            seq_id, residue_vec = aa_to_vector(pose, i)
            chain_id = pose.pdb_info().chain(i)
            if chain_id not in chains:
                chains[chain_id] = []
            if chain_id not in seqs:
                seqs[chain_id] = ''
            seqs[chain_id] += seq_id
            chains[chain_id].append(residue_vec)
        except AssertionError as e:
            print('Could not vectorize residue {}: {}'.format(i, str(e)))

    return seqs, chains


def chains_to_file(pdb_id, seqs, chains, out_folder=CHAINS_DIR):
    # Write chain files
    out_base = '{}/{}'.format(out_folder, pdb_id)

    try:
        os.makedirs(out_base)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise

    with open('{}/{}.fst'.format(out_base, pdb_id), 'w') as f:
        fasta = ''.join(map(lambda kv: '>{}:{}|PDBID|CHAIN|SEQUENCE\n{}\n'.format(pdb_id, kv[0], kv[1]), seqs.items()))
        f.write(fasta)

    for i, (chain_id, chain) in enumerate(chains.items()):
        out_path = '{}/{}_{}.npy'.format(out_base, pdb_id, chain_id)
        np.save(out_path, np.array(chain))


def process_pdb_list(files):
    for f in files:
        pdb_id = os.path.splitext(f)[0]
        pdb_path = os.path.join(PDB_DIR, f)
        if os.path.isdir(os.path.join(CHAINS_DIR, pdb_id)):
            print('{} already processed'.format(pdb_path))
        else:
            try:
                seqs, chains = pdb_to_chains(pdb_path)
                chains_to_file(pdb_id, seqs, chains)
                print('Processed {}'.format(pdb_path))
            except RuntimeError as e:
                print('Unable to process {}: {}'.format(pdb_id, str(e)))


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

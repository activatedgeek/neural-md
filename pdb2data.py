import os
import numpy as np
import pyrosetta
pyrosetta.init()

PDB_DIR = 'data/pdb'
CHAINS_DIR = 'data/chains'

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'U', 'G', 'P',
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
        # @TODO add max number of chi's
        *residue.chi()
    ]
    # normalize angles (-PI to PI)
    angles = list(map(lambda a: a / 180.0, angles))

    # @TODO add max number of atoms
    # atom coordinates, concatenation of x,y,z of each atom
    coords = []
    for atom in residue.atoms():
        coords.extend(atom.xyz())

    one_hot_type = [0] * len(RESIDUE_LETTERS)
    one_hot_type[RESIDUE_LETTERS.index(letter)] = 1

    return letter, [*coords, *angles, *one_hot_type]


def chains_to_file(pdb_path, out_folder):
    """

    :param pdb_path: Path to PDB file
    :param out_folder: Path to output folder, where a folder with the PDB filename will be created
    :return:
    """
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


def main():
    for f in os.listdir(PDB_DIR):
        if '.pdb' in f:
            chains_to_file(os.path.join(PDB_DIR, f), CHAINS_DIR)
        break


if __name__ == '__main__':
    main()

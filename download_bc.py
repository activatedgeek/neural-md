import os
import requests

CLUSTER_FILE = 'data/bc-70.out'

# Maximum number of clusters to process
MAX_CLUSTERS = 4


def download_pdb(pdb_id, target_dir='data/pdb'):
    path = '{}/{}.pdb'.format(target_dir, pdb_id)
    if os.path.isfile(path):
        print('{} already exists.'.format(path))
        return 0

    url = 'https://files.rcsb.org/download/{}.pdb'.format(pdb_id)

    r = requests.get(url)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print('Saved {}.'.format(path))
    return 1


def main():
    with open(CLUSTER_FILE, 'r') as f:
        c = 0
        for line in f:
            c += 1

            new_pdb_count = 0
            chain_list = line.split()
            for chain in chain_list:
                pdb_id = chain.split('_')[0]
                new_pdb_count += download_pdb(pdb_id)

            print('Cluster {}: {} new PDBs downloaded from {} chains'.format(c, new_pdb_count, len(chain_list)))

            if c >= MAX_CLUSTERS:
                break


if __name__ == '__main__':
    main()

import os
import requests
import threading

CLUSTER_FILE = 'data/bc-70.out'

# Maximum number of clusters to process
MAX_CLUSTERS = 4
MAX_THREADS = 4


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


class PDBDownloaderThread(threading.Thread):
    def __init__(self, id, chain_list):
        super(PDBDownloaderThread, self).__init__()

        self.id = id
        self.chain_list = chain_list

    def run(self):
        new_pdb_count = 0
        for chain in self.chain_list:
            pdb_id = chain.split('_')[0]
            new_pdb_count += download_pdb(pdb_id)

        print('Cluster {}: {} new PDBs downloaded from {} chains'.format(self.id, new_pdb_count, len(self.chain_list)))


def main():
    threads = []
    with open(CLUSTER_FILE, 'r') as f:
        c = 0
        for line in f:
            threads.append(PDBDownloaderThread(c, line.split()))

            c += 1
            if c >= MAX_CLUSTERS:
                break

    # A crude thread-limiter to download all PDBs faster
    join_list = []
    for id, t in enumerate(threads):
        t.run()
        join_list.append(id)

        if id > 0 and id % MAX_THREADS == 1:
            for idx in join_list:
                if threads[idx].isAlive():
                    threads[idx].join()
            join_list = []


if __name__ == '__main__':
    main()

from esm.models.esmc import ESMC
from esm.sdk.api import *
import os
import torch
import pandas as pd
import h5py
import numpy as np
import pickle
from tqdm import tqdm

os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = ESMC.from_pretrained("esmc_300m", device=device)

# max context 2048
all_amino_acid_number = {'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
                         'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
                         'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
                         'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
                         '_': 32}

def esm_encoder_seq(seq, pad_len):
    s = [all_amino_acid_number[x] for x in seq]
    while len(s) < pad_len:
        s.append(1)
    s.insert(0, 0)
    s.append(2)
    return torch.tensor(s)


def get_esm_embedding(seq):
    # str -> ndarray [960,]
    protein_tensor = ESMProteinTensor(sequence=esm_encoder_seq(seq, len(seq)).to(device))
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    esm_embedding = logits_output.embeddings
    assert isinstance(esm_embedding, torch.Tensor)
    return esm_embedding

def main():
    tags = ['train','test','val']
    if not os.path.exists('./esm_output'):
        os.mkdir('./esm_output', )
    for tag in tags:
        if not os.path.exists('./esm_output/' + tag):
            os.mkdir('./esm_output/{}'.format(tag))

        df = pd.read_csv('./data_csv/{}.csv'.format(tag),index_col='PDB')
        pdbs = np.unique(df.index)
        for i in tqdm(range(len(pdbs))):
            chains = np.unique(df.loc[pdbs[i]]['CHAIN'])
            with h5py.File('./data/{}/{}.h5'.format(tag,pdbs[i]), 'r') as fh5:
                sequences_dict = pickle.loads(fh5['sequences'][()])
            esm_embedding = {k:get_esm_embedding(j) for k,j in sequences_dict.items() if k in chains}
            for j,k in esm_embedding.items():
                torch.save(k, './esm_output/{}/{}_{}.pt'.format(tag,pdbs[i],j))

if __name__ == '__main__':
    main()



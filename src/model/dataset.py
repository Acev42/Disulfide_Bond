from torch.utils.data import Dataset,DataLoader
import pandas as pd
import h5py
import numpy as np
import torch
import pickle
import os
legal_res = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

def load_h5_file(h5_path,pdb,chain):
    try:
        with h5py.File(h5_path,'r') as hf:
            coords_dict = pickle.loads(hf['coordinates'][()])
            coord_list = coords_dict[chain]
            res_ids = np.array([i['res_id'] for i in coord_list])
            coord = np.array([i['coord'] for i in coord_list])
            res_names = np.array([i['res_name'] for i in coord_list])
            incomplete_res_ind = []
            embedding_mask = []
            for i,j in enumerate(res_names):
                if  j not in legal_res:
                    incomplete_res_ind.append(i)
                    if j == 'incomplete':
                        embedding_mask.append(i)
        return coord,res_ids,incomplete_res_ind,embedding_mask
    except Exception as e:
        print(e)

class DSDataset(Dataset):
    def __init__(self,
                 tag=None,csv_dir=None,predict=False,esm_dir=None,h5_dir=None
                 ):
        super().__init__()

        self.tag = tag
        self.predict = predict
        if predict:
            self.esm_dir = esm_dir
            self.h5_dir = h5_dir
        if csv_dir is None:
            self.csv_dir = f'./data_csv/{tag}.csv'
        else:
            self.csv_dir = csv_dir
        self.df = pd.read_csv(self.csv_dir,index_col='PDB',keep_default_na=False)
        self.pdb_ids = np.unique(self.df.index)


        chains_dict ={}
        for PDB in self.pdb_ids:
            chains_dict[PDB] = self.df.loc[PDB,'CHAIN']
        seq_list = []
        for pdb,chains in chains_dict.items():
            for chain in chains:
                sample_dict = {'PDB':pdb,'CHAIN':chain}
                if sample_dict not in seq_list:
                    seq_list.append(sample_dict)
        self.seqs_df = pd.DataFrame(seq_list)

    def load_embedding(self,pt_path):
        embedding = torch.load(pt_path)
        return embedding

    def __getitem__(self,index):
        seq = self.seqs_df.loc[index]
        pdb = seq.loc['PDB']
        chain = seq.loc['CHAIN']
        if not self.predict:
            h5_path = f'./data/{self.tag}/{pdb}.h5'
            esm_path = f'./esm_output/{self.tag}/{pdb}_{chain}.pt'
        else:
            h5_path = os.path.join(self.h5_dir, f'{pdb}.h5')
            esm_path  = os.path.join(self.esm_dir, f'{pdb}_{chain}.pt')
        coord,res_id,incomplete_res_index,embedding_mask = load_h5_file(h5_path,pdb,chain)
        embedding = self.load_embedding(esm_path)
        # n_prot, n_res+2, 960
        embedding = embedding[0, 1:-1, :]

        if len(incomplete_res_index) > 0 :
            mask = torch.ones(coord.shape[0], dtype=torch.bool)
            mask[incomplete_res_index] = False
            coord = coord[mask]
            res_id = res_id[mask]
        if len(embedding_mask) > 0 :
            mask = torch.ones(len(embedding), dtype=torch.bool)
            mask[embedding_mask] = False
            embedding = embedding[mask]
        if not self.predict:
            sample_df = self.df.loc[pdb]
            chain_mask = sample_df['CHAIN'] == chain
            sample_list = []
            label_list = []
            if type(chain_mask) != bool:
                sample_df = sample_df[chain_mask]
                for chain, res1, res2, label in sample_df.itertuples(index=0):
                    sample_list.append([res1, res2])
                    label_list.append(label)
            else:
                sample_list.append([sample_df['RES1'], sample_df['RES2']])
                label_list.append(sample_df['LABEL'])
        else:
            label_list, sample_list = [],[]

        return {'pdb':pdb,'chain':chain,'coord':coord,'embedding':embedding,
                'res_id':res_id,'labels':label_list,'samples':sample_list}


    def __len__(self):
        return len(self.seqs_df)
    def collate_fn(self,batch):
        batch = batch[0]
        coor = torch.tensor(batch['coord'],dtype=torch.float32)
        res_id = torch.tensor(batch['res_id'],dtype=torch.int64)
        embedding = torch.tensor(batch['embedding'],dtype=torch.float32)
        n_res = len(batch['coord'])
        connect = np.ones((n_res, 2), dtype=int) #whether the residue has left neighbor res or right neighbor res
        connect[0, 0] = 0
        connect[-1, 1] = 0
        connect = torch.tensor(connect, dtype=torch.int64)
        if not self.predict:
            labels = torch.tensor(batch['labels'],dtype=torch.float32)
            # [0,1,0,0...]
            samples = torch.tensor(batch['samples'],dtype=torch.int64)
            # [[res1,res2],]
        else:
            labels,samples = [],[]
        return {'coord':coor,'embedding':embedding,'res_id':res_id,
                'connect':connect,'pdb':batch['pdb'],'chain':batch['chain'],
                'label':labels,'sample':samples,'n_res':n_res}





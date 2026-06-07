import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.ESM_mapping import get_esm_embedding
from src.model.DSmodel import DS
from src.model.dataset import DSDataset

from biotite.structure.io import pdbx
from biotite.structure.io import pdb
import h5py
import pickle
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import argparse
from shutil import rmtree
from typing import Dict, List, Optional

alpha_3 = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
           'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
           'TRP', 'TYR', 'GAP']
alpha_1 = list("ACDEFGHIKLMNPQRSTVWY_")
aa_3_N = {a: n for n, a in enumerate(alpha_3)}
aa_3_1 = {a: b for a, b in zip(alpha_3, alpha_1)}
aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
aa_1_N = {a: n for n, a in enumerate(alpha_1)}


class ProteinProcessor:
    def __init__(self, input_dir: str, output_dir:str, index:str, pdb_id:str, save:str, ignore:str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.h5_dir = os.path.join(output_dir, "h5file")
        self.esm_dir = os.path.join(output_dir, "esm")
        self.csv_dir = os.path.join(output_dir, "csv")
        self.results_dir = os.path.join(output_dir, f"{index}_{pdb_id}_results.csv")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.save = save
        self.ignore = ignore

        for dir_path in [self.h5_dir, self.esm_dir, self.csv_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_sequence(self, atom_array) -> Dict[str, str]:
        seq = {}
        amino_acids = atom_array[np.isin(atom_array.res_name, alpha_3)]
        chain_ids = np.unique(amino_acids.chain_id)
        for chain in chain_ids:
            tgt_chain = amino_acids[amino_acids.chain_id == chain]
            residues = []
            seen_residues = set()

            for res_id, res_name in zip(tgt_chain.res_id, tgt_chain.res_name):
                if res_id not in seen_residues:
                    seen_residues.add(res_id)
                    residues.append(aa_3_1[res_name])

            seq[chain] = ''.join(residues)
        return seq


    def build_Cb(self, cacoor, ncoor, ccoor):

        b = cacoor - ncoor
        c = ccoor - cacoor
        a = np.cross(b, c)
        cbcoor = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + cacoor
        return cbcoor

    def grep_backbone(self,atom_array, chains):

        C_mask = atom_array.atom_name == 'C'
        Ca_mask = atom_array.atom_name == 'CA'
        N_mask = atom_array.atom_name == 'N'
        Cb_mask = atom_array.atom_name == 'CB'
        O_mask = atom_array.atom_name == 'O'

        backbone_mask = C_mask | Ca_mask | N_mask | O_mask | Cb_mask
        backbone_atom_array = atom_array[backbone_mask]
        chain_dict = {}
        for chain in chains:
            tgt_chain = backbone_atom_array[backbone_atom_array.chain_id == chain]
            residues = []
            residue_ids = []
            seen_residues = set()

            for res_id, res_name in zip(tgt_chain.res_id, tgt_chain.res_name):
                if res_id not in seen_residues:
                    seen_residues.add(res_id)
                    residue_ids.append(res_id)

            for i in residue_ids:
                atoms_in_res = tgt_chain[tgt_chain.res_id == i]
                NCCOC_coord = np.zeros((5, 3))
                residue = {}
                residue['res_id'] = i
                residue['chain_id'] = chain
                residue['res_name'] = 0
                has_required = {'N': False, 'CA': False, 'C': False, 'O': False, 'CB': False}
                for atom in atoms_in_res:
                    if atom.atom_name == 'N':
                        NCCOC_coord[0, :] = atom.coord
                        has_required['N'] = True
                    elif atom.atom_name == 'CA':
                        NCCOC_coord[1, :] = atom.coord
                        has_required['CA'] = True
                    elif atom.atom_name == 'C':
                        NCCOC_coord[2, :] = atom.coord
                        has_required['C'] = True
                    elif atom.atom_name == 'O':
                        NCCOC_coord[3, :] = atom.coord
                        has_required['O'] = True
                    elif atom.atom_name == 'CB':
                        NCCOC_coord[4, :] = atom.coord
                        has_required['CB'] = True
                    if not residue['res_name']:
                        residue['res_name'] = atom.res_name
                if (not has_required['CB']) and has_required['C'] and has_required['N'] and has_required['CA']:
                    NCCOC_coord[4, :] = self.build_Cb(NCCOC_coord[1, :], NCCOC_coord[0, :], NCCOC_coord[2, :])
                residue['coord'] = NCCOC_coord
                if not (has_required['C'] and has_required['N'] and has_required['CA'] and has_required['O']):
                    residue['res_name'] = 'incomplete'
                residues.append(residue)
            chain_dict[chain] = residues

        return chain_dict

    def save_pdb_h5(self, pdb_id: str, sequences: Dict, backbone: Dict,
                    save_dir: Optional[str] = None) -> str:

        if save_dir is None:
            save_dir = self.h5_dir

        os.makedirs(save_dir, exist_ok=True)
        chain_ids = list(sequences.keys())
        lengths = [len(seq) for seq in sequences.values()]


        byte_chains = np.void(pickle.dumps(chain_ids))
        byte_len = np.void(pickle.dumps(lengths))
        byte_backbone = np.void(pickle.dumps(backbone))
        byte_sequences = np.void(pickle.dumps(sequences))

        
        h5_path = os.path.join(save_dir, f'{pdb_id}.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('chain_ids', data=byte_chains)
            f.create_dataset('lengths', data=byte_len)
            f.create_dataset('sequences', data=byte_sequences)
            f.create_dataset('coordinates', data=byte_backbone)
            f.attrs['PDB ID'] = pdb_id

        print(f"Saved H5 file: {h5_path}")
        return h5_path

    def save_esm_embeddings(self, pdb_id: str, chains: List[str],
                            h5_path: str, save_dir: Optional[str] = None):

        if save_dir is None:
            save_dir = self.esm_dir

        os.makedirs(save_dir, exist_ok=True)

        with h5py.File(h5_path, 'r') as fh5:
            sequences_dict = pickle.loads(fh5['sequences'][()])

        for chain_id in chains:
            if chain_id in sequences_dict:
                sequence = sequences_dict[chain_id]
                try:
                    embedding = get_esm_embedding(sequence)
                    esm_path = os.path.join(save_dir, f'{pdb_id}_{chain_id}.pt')
                    torch.save(embedding, esm_path)
                    print(f"Saved ESM embedding: {esm_path}")
                except Exception as e:
                    print(f"Error generating ESM embedding for {pdb_id}_{chain_id}: {e}")


    def process_single_pdb(self, pdb_id, target_chains,file_type) :

        if file_type == 'CIF':
            cif_path = self.input_dir
            cif_file = pdbx.CIFFile.read(cif_path)
            atom_array = pdbx.get_structure(cif_file, model=1,
                                            use_author_fields=False, altloc='occupancy')
        elif file_type == 'PDB':
            pdb_path = self.input_dir
            pdb_file = pdb.PDBFile.read(pdb_path)
            atom_array = pdb.get_structure(pdb_file, model=1, altloc='occupancy')

        atom_array = atom_array[~atom_array.hetero]
        sequences = self.get_sequence(atom_array)

        if target_chains is None:
            target_chains = list(sequences.keys())
        else:
            available_chains = list(sequences.keys())
            target_chains = [chain for chain in target_chains if chain in available_chains]
            if not target_chains:
                raise ValueError(f"None of the specified chains found. Available chains: {available_chains}")

        backbone = self.grep_backbone(atom_array, target_chains)

        sequences = {chain: seq for chain, seq in sequences.items() if chain in target_chains}

        return pdb_id, sequences, backbone, target_chains


    def get_res_names_and_ind_dict(self,h5_path,chain):
        with h5py.File(h5_path, 'r') as fh5:
            coords_dict = pickle.loads(fh5['coordinates'][()])
            coord_list = coords_dict[chain]
            res_ids = np.array([i['res_id'] for i in coord_list])
            id_to_ind = {j:i for i,j in enumerate(res_ids)}
            res_names = np.array([i['res_name'] for i in coord_list])
        return res_names, id_to_ind


    def predict(self, csv_path: str,esm_dir,h5_dir,model_path: Optional[str] = None) -> Dict:

        dataset = DSDataset(csv_dir=csv_path,esm_dir=esm_dir,h5_dir=h5_dir,predict=True)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=dataset.collate_fn)

        
        if model_path and os.path.exists(model_path):
            model = DS.load_from_checkpoint(model_path,strict=False)
        else:
            print('model path does not exist')
            raise OSError

               
        trainer = pl.Trainer(
            devices='auto',
            precision='64',
            accelerator='cuda',
            inference_mode=True,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            default_root_dir=self.logs_dir
        )

        predictions = trainer.predict(model, data_loader)

        all_results = {}
        for batch_result in predictions:
            if batch_result:
                pdb_id, chain, output = batch_result
                if pdb_id not in all_results:
                    all_results[pdb_id] = {}
                if chain not in all_results[pdb_id]:
                    all_results[pdb_id][chain] = []
                h5_path  = os.path.join(h5_dir, f'{pdb_id}.h5')
                all_res_names, id2ind = self.get_res_names_and_ind_dict(h5_path, chain)
                for (res1, res2), prob in output.items():
                    if abs(res1-res2)>3:
                        res1_name = all_res_names[id2ind[res1]]
                        res2_name = all_res_names[id2ind[res2]]
                        if self.ignore and res1_name == 'CYS' and res2_name == 'CYS':
                            continue
                        all_results[pdb_id][chain].append({
                                'residue1 id': res1,
                                'residue1 name': res1_name,
                                'residue2 id': res2,
                                'residue2 name': res2_name,
                                'probability': float(prob)
                            })
            all_dsbonds_data = []

        for pdb_id, chains_data in all_results.items():
            for chain, dsbonds in chains_data.items():
                for dsbond in dsbonds:
                    dsbond_record = dsbond.copy()
                    dsbond_record['pdb_id'] = pdb_id
                    dsbond_record['chain'] = chain
                    all_dsbonds_data.append(dsbond_record)

        csv_output = self.results_dir
        if all_dsbonds_data:
            dsbond_df = pd.DataFrame(all_dsbonds_data)
            dsbond_df.to_csv(csv_output, index=False)

        if not self.save:
            rmtree(self.h5_dir)
            rmtree(self.esm_dir)
            rmtree(self.csv_dir)
            rmtree(self.logs_dir)
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Disulfide Prediction')
    parser.add_argument('--index',type=str,default=None)
    parser.add_argument('--threshold', type=str, default='0.5',)
    parser.add_argument('--input', type=str, default='./input/')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--pdb', type=str,default=None,
                        help='PDB ID for assigned mode')
    parser.add_argument('--chain', type=str,default=None,
                        help='target chain id (label chain id,separated by comma)')
    parser.add_argument('--filetype', type=str,
                        choices=['PDB','CIF'],default='PDB')
    parser.add_argument('--model', type=str, default='./src/model/Disulfide_searching.ckpt',
                        help='the path of model file')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--ignore',type=bool,default=True)

    args = parser.parse_args()
    if args.pdb is None:
        args.pdb = args.input[-8:-4]
    processor = ProteinProcessor(args.input, args.output, args.index, args.pdb,args.save,args.ignore)
    pdb_id, sequences, backbone, processed_chains = processor.process_single_pdb(args.pdb, args.chain, file_type=args.filetype)
    h5_path = processor.save_pdb_h5(pdb_id, sequences, backbone)
    processor.save_esm_embeddings(pdb_id, processed_chains, h5_path)
    csv_path = os.path.join(processor.csv_dir, f'{pdb_id}.csv')
    pd.DataFrame({'PDB': [pdb_id for i in processed_chains], 'CHAIN': processed_chains}).to_csv(csv_path, index=False)
    processor.predict(csv_path, esm_dir=processor.esm_dir,h5_dir= processor.h5_dir,model_path=args.model)


if __name__ == "__main__":
    main()
# example python run.py --model "./Disulfide_searching(pos_weight_72)"

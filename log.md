# Model workflow
- Read `.pdb` or `.cif` file and extract amino acids sequences from specific chains.
- Identify heavy atoms in bacbbone (i.e. C,CB,O,N,CA) and write down their coordinations `[n,5,3]`.  (`n` stands for the number of AAS)
- AAs sequences `[n,]` -> ESM-C -> embedding `[n,960,]`
- Embedding, coordination -> MPNNs -> disulfide probabilities `[n,n]` (the probability of each pair of AAs to form disulfide)
- Indentify AA pairs, probability of which exceeds threshold 

# Dataset
-`pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2025_06_02_chains11750` 25% clustered, resolution 0-0.25 A, R_factor <0.3, length 40-10000
- Download `.cif` file from PDBbank
- Identify all the disulfide bonds in primitive dataset, and write down the distances of Cys Ca atoms. Get the average disulfide bond length. Proteins with disulfide bond are processed as `Positive samples`.
- Based on the average bond length calculated, find AA pairs whose Ca distance approach average bond length. These proteins are processed as `Negative samples`

# MPNN architecture
-Node features: 
-Edge features:
-Message transpassing methods:

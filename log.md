# Model workflow
- Read `.pdb` or `.cif` file and extract amino acids sequences from specific chains.
- Identify heavy atoms in backbone (i.e. C,O,N,Ca and Cb) and write down their coordinations `[n,5,3]`.  (`n` stands for the number of AAs)
- AAs sequences `[n,]` -> ESM-C -> embedding `[n,960,]`
- Embedding, coordination -> MPNNs feature construction
- Graph building (DSgraphlayer * N)
- Global attention
- Edge classification-> disulfide probabilities `[num_edges,1]` (the probability of each pair of AAs to form disulfide)
- Indentify AA pairs, probability of which exceeds threshold 

# Dataset
-`pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2025_06_02_chains11750` 25% clustered, resolution 0-0.25 A, R_factor <0.3, length 40-10000
- Download `.cif` file from PDBbank
- Identify all the disulfide bonds in primitive dataset, and write down the distances of Cys Ca atoms. Get the average disulfide bond length.
- Positive samples：(inedx_Cys_i, index_Cys_j, 1) forming native disulfide bond (ProteinMPNN is utilized to convert Cys into other possible AAs)
- Negative samples:(index_AA_i, index_AA_j, 0) Candidate residue pairs not forming disulfide bond but Ca distance close to disulfide bond

# MPNN architecture
Edge construction: Ca distance threshold 10A

-Node features: `[,4*3]` (coordination of Cb,O,N,C relative to Ca) + `[,4*rbfn]` (RBF encoding of distance from Cb,O,N,C to Ca ) + `[n,960]`embedding + `[,14]`(dihedrals and bond angle)

-Edge features: `[,rbfn]`(RBF encoding of Ca distance) + `[,3] `(edge vector)+`[,30*rbfN_coor]` (RBF encoding of relative coordinates of 10 backbone atoms) + `[,65]`(embedding of Sequence separation)

-Message transpassing methods: 

`message = MLP(target node feature, edge feature, geometric feature)`

`attention = MLP(source node, target node, edge feature, geometric feature)`

`out = message * attention`

`scatter(out,index=edge_index[0],reduce='sum')`

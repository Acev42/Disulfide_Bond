import torch
from einops import rearrange, reduce, repeat

class rigid_transform:
    def __init__(self, T, R, R_inv=None):
        #T: n, 3  translation
        #R: n, 3, 3 rotation
        assert len(T.shape) == 2
        assert len(R.shape) == 3
        assert T.shape[-1] == 3
        assert R.shape[-1] == 3
        assert R.shape[-2] == 3
        assert T.shape[0] == R.shape[0]
        self.T = T
        self.R = R
        if R_inv is None:
            self.R_inv = torch.linalg.inv(R)
        else:
            self.R_inv = R_inv

        self.shape = {"T":T.shape, "R":R.shape}

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, i):
        # get the coord of specific residue
        if not torch.is_tensor(i):
            index = torch.tensor([i], dtype=torch.int)
        else:
            index = i
        return rigid_transform(T=self.T[index], R=self.R[index], R_inv=self.R_inv[index])


    def compose(self, transform):
        T_new = torch.einsum('...i...ij->...j',transform.T, self.R) + self.T
        R_new = torch.einsum('...ij...jl->...il', self.R,transform.R)
        return rigid_transform(T_new, R_new)

    def apply_atom(self, coor): #one atom per residue
        assert coor.shape == self.T.shape
        return torch.einsum('ni,nij->nj', coor, self.R) + self.T

    def apply_residue(self, coor): #>1 atom per residue
        assert coor.shape[0] == self.T.shape[0]
        assert coor.shape[2] == self.T.shape[1]
        assert len(coor.shape) == 3
        return torch.einsum('nmi,nij->nmj', coor, self.R) + self.T[:,None,:]

    def reverse_apply_atom(self, coor):
        assert coor.shape == self.T.shape
        return torch.einsum('ni,nij->nj', coor-self.T, self.R_inv)

    def reverse_apply_residue(self, coor): #>1 atom per residue
        #coor: nres natom 3
        #T: nres 3
        assert coor.shape[0] == self.T.shape[0]
        assert len(coor.shape) == 3 
        return torch.einsum('nmi,nij->nmj', coor-self.T[:,None,:], self.R_inv)





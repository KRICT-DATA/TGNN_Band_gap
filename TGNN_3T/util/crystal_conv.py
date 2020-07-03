import numpy
import random
import pandas
import torch
from mendeleev import get_table
from mendeleev import element
from pymatgen.core.structure import Structure
from sklearn import preprocessing
from util.math import even_samples
from util.math import RBF
from util.data import AtomwiseCrystal

atom_feat_names = ['covalent_radius_bragg',]

atom_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
# atom_feat_names = ['period', 'atomic_number', 'atomic_weight', 'vdw_radius', 'en_pauling']
mat_atom_feats = None
num_atom_feats = 0
num_bond_feats = 8
mean = None
std = None


def load_mat_atom_feats():
    tb_atom_feats = get_table('elements')
    atom_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[atom_feat_names]))[:96, :]
    ion_engs = numpy.zeros((atom_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies

        if 1 in ion_eng:
            ion_engs[i, 0] = ion_eng[1]
        else:
            ion_engs[i, 0] = 0

    global mat_atom_feats, num_atom_feats
    mat_atom_feats = preprocessing.scale(numpy.hstack((atom_feats, ion_engs)))
    num_atom_feats = mat_atom_feats.shape[1]


def load_dataset(path, target_idx, ref_idx, train_ratio=0.6, val_ratio=0.2, radius=4):
    list_crys = list()
    id_target = numpy.array(pandas.read_csv(path + '/id_target_temp.csv'))
    num_train_ins = int(id_target.shape[0] * train_ratio)
    num_opt_ins = num_train_ins + int(id_target.shape[0] * val_ratio)

    for i in range(0, id_target.shape[0]):
        crys = read_cif(path, str(id_target[i, 0]), id_target[i, target_idx], radius, id_target[i, ref_idx], i)

        if crys is not None:
            list_crys.append(crys)

        if (i + 1) % 100 == 0:
            print('Complete loading {:}th crystal.'.format(i + 1))

    random.shuffle(list_crys)

    if val_ratio > 0:
        return list_crys[:num_train_ins], list_crys[num_train_ins:num_opt_ins], list_crys[num_opt_ins:]
    else:
        return list_crys[:num_train_ins], list_crys[num_train_ins:]


def read_cif(path, m_id, target, radius, ref_feat, id):
    crys = Structure.from_file(path + '/' + m_id + '.cif')
    atoms = crys.atomic_numbers
    list_nbrs = crys.get_all_neighbors(radius, include_index=True)
    rbf_means = even_samples(5, radius, num_bond_feats)
    pairs = get_pairs(atoms, list_nbrs, rbf_means)
    triplets = get_triplets(atoms, list_nbrs, rbf_means)

    if pairs is None or triplets is None:
        return None

    pairs = torch.tensor(pairs, dtype=torch.float)
    triplets = torch.tensor(triplets, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.float).view(-1, 1)
    ref_feat = torch.tensor(ref_feat, dtype=torch.float).view(-1, 1)

    return AtomwiseCrystal(pairs, triplets, target, ref_feat, id, m_id)


def get_pairs(atoms, list_nbrs, means):
    pairs = list()

    for i in range(0, len(list_nbrs)):
        atom_feats1 = mat_atom_feats[atoms[i]-1, :]
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            atom_feats2 = mat_atom_feats[atoms[nbrs[j][2]-1], :]
            bond_feats = RBF(numpy.full(means.shape[0], nbrs[j][1]), means, beta=2)
            pairs.append(numpy.hstack([atom_feats1, atom_feats2, bond_feats]))

    if len(pairs) == 0:
        return None

    return numpy.vstack(pairs)


def get_triplets(atoms, list_nbrs, means):
    triplets = list()

    for i in range(0, len(list_nbrs)):
        atom_feats1 = mat_atom_feats[atoms[i]-1, :]
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            atom_feats2 = mat_atom_feats[atoms[nbrs[j][2]-1], :]
            bond_feats1 = RBF(numpy.full(means.shape[0], nbrs[j][1]), means, beta=2)
            n_nbrs = list_nbrs[nbrs[j][2]]

            for k in range(0, len(n_nbrs)):
                atom_feats3 = mat_atom_feats[atoms[n_nbrs[k][2]-1], :]
                bond_feats2 = RBF(numpy.full(means.shape[0], n_nbrs[k][1]), means, beta=2)

                triplets.append(numpy.hstack([atom_feats1, atom_feats2, atom_feats3, bond_feats1, bond_feats2]))

    if len(triplets) == 0:
        return None

    return numpy.vstack(triplets)


load_mat_atom_feats()

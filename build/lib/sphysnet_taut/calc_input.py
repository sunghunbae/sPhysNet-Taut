import numpy as np
import torch
import torch_geometric

from scipy.spatial import Voronoi
from torch_geometric.data import Data

import os.path as osp
import os

from .gen_confs import get_low_energy_conf

_force_cpu = False


def get_coords(mol):
    conf = mol.GetConformer()
    coords = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append( [pos.x, pos.y, pos.z] )
    return np.array(coords)


def get_elements(mol):
    z = []
    for atom in mol.GetAtoms():
        z.append(atom.GetAtomicNum())
    return np.array(z)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _get_index_from_matrix(num, previous_num):
    """
    get the fully-connect graph edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    index = torch.LongTensor(2, num * num)
    index[0, :] = torch.cat([torch.zeros(num).long().fill_(i) for i in range(num)], dim=0)
    index[1, :] = torch.cat([torch.arange(num).long() for __ in range(num)], dim=0)
    mask = (index[0, :] != index[1, :])
    return index[:, mask] + previous_num


def cal_edge(R, N, prev_N, edge_index, cal_coulomb=True, short_range=True):
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
    :param short_range:
    :param cal_coulomb:
    :param prev_N:
    :param edge_index:
    :param R:
    :param N:
    :return:
    """
    if cal_coulomb:
        '''
        IMPORTANT: DO NOT use num(tensor) itself as input, which will be regarded as dictionary key in this function,
        use int value(num.item())
        Using tensor as dictionary key will cause unexpected problem, for example, memory leak
        '''
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), previous_num) for num, previous_num in zip(N, prev_N)], dim=-1)
        points1 = R[coulomb_index[0, :], :]
        points2 = R[coulomb_index[1, :], :]
        coulomb_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        coulomb_dist = torch.sqrt(coulomb_dist)

    else:
        coulomb_dist = None
        coulomb_index = None

    if short_range:
        short_range_index = edge_index
        points1 = R[edge_index[0, :], :]
        points2 = R[edge_index[1, :], :]
        short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        short_range_dist = torch.sqrt(short_range_dist)
    else:
        short_range_dist, short_range_index = None, None
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def scale_R(R):
    abs_min = torch.abs(R).min()
    while abs_min < 1e-3:
        R = R - 1
        abs_min = torch.abs(R).min()
    return R


def cal_msg_edge_index(edge_index):
    msg_id_1 = torch.arange(edge_index.shape[-1]).repeat(edge_index.shape[-1], 1)
    msg_id_0 = msg_id_1.t()
    source_atom = edge_index[0, :].repeat(edge_index.shape[-1], 1)
    target_atom = edge_index[1, :].view(-1, 1)
    msg_map = (source_atom == target_atom)
    result = torch.cat([msg_id_0[msg_map].view(1, -1), msg_id_1[msg_map].view(1, -1)], dim=0)
    return result


def voronoi_edge_index(R, boundary_factor, use_center):
    """
    Calculate Voronoi Diagram
    :param R: shape[-1, 3], the location of input points
    :param boundary_factor: Manually setup a boundary for those points to avoid potential error, value of [1.1, inf]
    :param use_center: If true, the boundary will be centered on center of points; otherwise, boundary will be centered
    on [0., 0., 0.]
    :return: calculated edge idx_name
    """
    R = scale_R(R)

    R_center = R.mean(dim=0) if use_center else torch.DoubleTensor([0, 0, 0])

    # maximum relative coordinate
    max_coordinate = torch.abs(R - R_center).max()
    boundary = max_coordinate * boundary_factor
    appended_R = torch.zeros(8, 3).double().fill_(boundary)
    idx = 0
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            for z_sign in [-1, 1]:
                appended_R[idx] *= torch.DoubleTensor([x_sign, y_sign, z_sign])
                idx += 1
    num_atoms = R.shape[0]

    appended_R = appended_R + R_center
    diagram = Voronoi(torch.cat([R, appended_R], dim=0), qhull_options="Qbb Qc Qz")
    edge_one_way = diagram.ridge_points
    edge_index_all = torch.LongTensor(np.concatenate([edge_one_way, edge_one_way[:, [1, 0]]], axis=0)).t()
    mask0 = edge_index_all[0, :] < num_atoms
    mask1 = edge_index_all[1, :] < num_atoms
    mask = mask0 & mask1
    edge_index = edge_index_all[:, mask]
    return edge_index

def sort_edge(edge_index):
    """
    sort the target of edge to be sequential, which may increase computational efficiency later on when training
    :param edge_index:
    :return:
    """
    arg_sort = torch.argsort(edge_index[1, :])
    return edge_index[:, arg_sort]


def mol_to_edge_index(mol):
    """
    Calculate edge_index(bonding edge) from rdkit.mol
    :param mol:
    :return:
    """
    bonds = mol.GetBonds()
    num_bonds = len(bonds)
    _edge_index = torch.zeros(2, num_bonds).long()
    for bond_id, bond in enumerate(bonds):
        _edge_index[0, bond_id] = bond.GetBeginAtomIdx()
        _edge_index[1, bond_id] = bond.GetEndAtomIdx()
    _edge_index_inv = _edge_index[[1, 0], :]
    _edge_index = torch.cat([_edge_index, _edge_index_inv], dim=-1)
    return _edge_index


def remove_bonding_edge(all_edge_index, bond_edge_index):
    """
    Remove bonding idx_name from atom_edge_index to avoid double counting
    :param all_edge_index:
    :param bond_edge_index:
    :return:
    """
    mask = torch.zeros(all_edge_index.shape[-1]).bool().fill_(False).type(all_edge_index.type())
    len_bonding = bond_edge_index.shape[-1]
    for i in range(len_bonding):
        same_atom = (all_edge_index == bond_edge_index[:, i].view(-1, 1))
        mask += (same_atom[0] & same_atom[1])
    remain_mask = ~ mask
    return all_edge_index[:, remain_mask]


def extend_bond(edge_index):
    """
    extend bond edge to a next degree, i.e. consider all 1,3 interaction as bond
    :param edge_index:
    :return:
    """
    n_edge = edge_index.size(-1)
    source = edge_index[0]
    target = edge_index[1]

    # expand into a n*n matrix
    source_expand = source.repeat(n_edge, 1)
    target_t = target.view(-1, 1)

    mask = (source_expand == target_t)
    target_index_mapper = edge_index[1].repeat(n_edge, 1)
    source_index_mapper = edge_index[0].repeat(n_edge, 1).t()

    source_index = source_index_mapper[mask]
    target_index = target_index_mapper[mask]

    extended_bond = torch.cat([source_index.view(1, -1), target_index.view(1, -1)], dim=0)
    # remove self to self interaction
    extended_bond = extended_bond[:, source_index != target_index]
    extended_bond = remove_bonding_edge(extended_bond, edge_index)
    result = torch.cat([edge_index, extended_bond], dim=-1)

    result = torch.unique(result, dim=1)
    return result


def my_pre_transform(data, edge_version, do_sort_edge, cal_efg, cutoff, boundary_factor, use_center, mol,
                     cal_3body_term, bond_atom_sep, record_long_range, type_3_body='B', extended_bond=False):
    """
    edge calculation
    atom_edge_index is non-bonding edge idx_name when bond_atom_sep=True; Otherwise, it is bonding and non-bonding together
    """
    edge_index = torch.zeros(2, 0).long()
    dist, full_edge, _, _ = cal_edge(data.pos, [data.N], [0], edge_index, cal_coulomb=True, short_range=False)
    dist = dist.cpu()
    full_edge = full_edge.cpu()

    if edge_version == 'cutoff':
        data.BN_edge_index = full_edge[:, (dist < cutoff).view(-1)]
    else:
        data.BN_edge_index = voronoi_edge_index(data.pos, boundary_factor, use_center=use_center)

    if record_long_range:
        data.L_edge_index = remove_bonding_edge(full_edge, data.BN_edge_index)

    '''
    sort edge idx_name
    '''
    if do_sort_edge:
        data.BN_edge_index = sort_edge(data.BN_edge_index)

    '''
    EFGs edge calculation
    '''
    if cal_efg:
        if edge_version == 'cutoff':
            dist, full_edge, _, _ = cal_edge(data.EFG_R, [data.EFG_N], [0], edge_index, cal_coulomb=True)
            data.EFG_edge_index = full_edge[:, (dist < cutoff).view(-1)].cpu()
        else:
            data.EFG_edge_index = voronoi_edge_index(data.EFG_R, boundary_factor, use_center=use_center)

        data.num_efg_edges = torch.LongTensor([data.EFG_edge_index.shape[-1]]).view(-1)

    if bond_atom_sep:
        '''
        Calculate bonding edges and remove those non-bonding edges which overlap with bonding edge
        '''
        if mol is None:
            print('rdkit mol file not given for molecule: {}, cannot calculate bonding edge, skipping this'.format(
                data.Z))
            return None
        B_edge_index = mol_to_edge_index(mol)
        if B_edge_index.numel() > 0 and B_edge_index.max() + 1 > data.N:
            raise ValueError('problematic mol file: {}'.format(mol))
        if B_edge_index.numel() > 0 and extended_bond:
            B_edge_index = extend_bond(B_edge_index)
        if B_edge_index.numel() > 0 and do_sort_edge:
            B_edge_index = sort_edge(B_edge_index)
        data.B_edge_index = B_edge_index
        try:
            data.N_edge_index = remove_bonding_edge(data.BN_edge_index, B_edge_index)
        except Exception as e:
            print("*"*40)
            print("BN: ", data.BN_edge_index)
            print("B: ", data.B_edge_index)
            from rdkit.Chem import MolToSmiles
            print("SMILES: ", MolToSmiles(mol))
            raise e
        _edge_list = []
        for bond_type in type_3_body:
            _edge_list.append(getattr(data, bond_type + "_edge_index"))
        _edge_index = torch.cat(_edge_list, dim=-1)
    else:
        _edge_index = data.BN_edge_index

    '''
    Calculate 3-atom term(Angle info)
    It ls essentially an "edge" of edge
    '''
    if cal_3body_term:

        atom_msg_edge_index = cal_msg_edge_index(_edge_index)
        if do_sort_edge:
            atom_msg_edge_index = sort_edge(atom_msg_edge_index)

        setattr(data, type_3_body + '_msg_edge_index', atom_msg_edge_index)

        setattr(data, 'num_' + type_3_body + '_msg_edge', torch.zeros(1).long() + atom_msg_edge_index.shape[-1])

    for bond_type in ['B', 'N', 'L', 'BN']:
        _edge_index = getattr(data, bond_type + '_edge_index', False)
        if _edge_index is not False:
            setattr(data, 'num_' + bond_type + '_edge', torch.zeros(1).long() + _edge_index.shape[-1])

    return data


def calc_data_for_predict(smi, num_confs=300):
    mol = get_low_energy_conf(smi, num_confs=num_confs)
    coords = get_coords(mol)
    elements = get_elements(mol)

    N = coords.shape[0]
    this_data = Data(pos = torch.as_tensor(coords, dtype=torch.double),
                      Z = torch.as_tensor(elements, dtype=torch.long),
                      N = torch.as_tensor(N, dtype=torch.long).view(-1),
                      BN_edge_index_correct = torch.tensor([0], dtype=torch.long),
                      batch = torch.tensor([0 for i in range(N)], dtype=torch.long))

    nthis_data = my_pre_transform( this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                    cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                    bond_atom_sep=False, record_long_range=True)
    return nthis_data

if __name__=="__main__":
    smi = "CCC"
    data = calc_data_for_predict(smi, num_confs=300)
    print( data )

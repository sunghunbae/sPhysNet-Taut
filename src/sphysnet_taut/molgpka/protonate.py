from sphysnet_taut.molgpka.predict_pka import predict
from copy import deepcopy
from rdkit import Chem

from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdmolops
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from itertools import combinations

import json
import numpy as np
import random
import os
import copy

def modify_mol(mol, acid_dict, base_dict):
    for at in mol.GetAtoms():
        idx = at.GetIdx()
        if idx in set(acid_dict.keys()):
            value = acid_dict[idx]
            nat = at.GetNeighbors()[0]
            nat.SetProp("ionization", "A")
            nat.SetProp("pKa", str(value))
        elif idx in set(base_dict.keys()):
            value = base_dict[idx]
            at.SetProp("ionization", "B")
            at.SetProp("pKa", str(value))
        else:
            at.SetProp("ionization", "O")
    return mol


def get_pKa_data(mol, ph, tph):
    stable_data, unstable_data = [], []
    for at in mol.GetAtoms():
        props = at.GetPropsAsDict()
        acid_or_basic = props.get('ionization', False)
        pKa = float(props.get('pKa', False))
        idx = at.GetIdx()
        if acid_or_basic == "A":
            if pKa < ph - tph:
                stable_data.append( [idx, pKa, "A"] )
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append( [idx, pKa, "A"] )
        elif acid_or_basic == "B":
            if pKa > ph + tph:
                stable_data.append( [idx, pKa, "B"] )
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append( [idx, pKa, "B"] )
        else:
            continue
    return stable_data, unstable_data


def get_neighbor_hydrogen(at):
    nats = at.GetNeighbors()
    h_nat_idxs = []
    for at in nats:
        if at.GetSymbol() == "H":
            h_nat_idxs.append(at.GetIdx())
    h_nat_idxs.sort(reverse=True)
    return h_nat_idxs


def remove_atom(idx, mol):
    emol = Chem.EditableMol(mol)
    emol.RemoveAtom(idx)
    nmol = emol.GetMol()
    return nmol


def modify_acid(at, mol):
    at.SetFormalCharge(-1)
    h_nat_idxs = get_neighbor_hydrogen( at )
    remove_h_idx = h_nat_idxs[0]
    nmol = remove_atom(remove_h_idx, mol)
    return nmol


def add_atom(at, mol):
    emol = Chem.EditableMol(mol)
    h_atom = Chem.Atom(1)
    h_idx = emol.AddAtom(h_atom)
    emol.AddBond(at.GetIdx(), h_idx, order=Chem.rdchem.BondType.SINGLE)
    new_mol = emol.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def modify_base(at, mol):
    at.SetFormalCharge(1)
    new_mol = add_atom(at, mol)
    return new_mol


def modify_stable_pka(new_mol, stable_data):
    for pka_data in stable_data:
        idx, pka, acid_or_basic = pka_data
        at = new_mol.GetAtomWithIdx(idx)
        if acid_or_basic == "A":
            new_mol = modify_acid(at, new_mol)
        elif acid_or_basic == "B":
            new_mol = modify_base(at, new_mol)
    return new_mol


def modify_unstable_pka(mol, unstable_data, i):
    combine_pka_datas = list(combinations(unstable_data, i))
    new_unsmis = []
    for pka_datas in combine_pka_datas:
        new_mol = deepcopy(mol)
        if len(pka_datas) == 0:
            continue
        for pka_data in pka_datas:
            idx, pka, acid_or_basic = pka_data
            at = new_mol.GetAtomWithIdx(idx)
            if acid_or_basic == "A":
                new_mol = modify_acid(at, new_mol)
            elif acid_or_basic == "B":
                new_mol = modify_base(at, new_mol)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_unsmis.append(smi)
    return new_unsmis


def protonate_mol(smi, ph, tph):
    omol = Chem.MolFromSmiles(smi)
    obase_dict, oacid_dict, omol = predict(omol)
    #print(oacid_dict)
    mc = modify_mol(omol, oacid_dict, obase_dict)
    stable_data, unstable_data = get_pKa_data(mc, ph, tph)
    
    new_smis = []
    n = len(unstable_data)
    if n == 0:
        new_mol = deepcopy(mc)
        new_mol = modify_stable_pka(new_mol, stable_data)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol)))
        new_smis.append(smi)
    else:
        for i in range(n + 1):
            new_mol = deepcopy(mc)
            modify_stable_pka(new_mol, stable_data)
            if i == 0:
                new_smis.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))))
            new_unsmis = modify_unstable_pka(new_mol, unstable_data, i)
            new_smis.extend(new_unsmis)
    return new_smis


if __name__=="__main__":
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    smi = "CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O"
    smi = "Nc1cc(C(F)(F)F)c(-c2cc(N3CCCC3)nc(N3CCOCC3)n2)cn1"
    smi = "O=C(O)c1cncc(O)n1"
    smi = "O=C(O)c1cncc(=O)[nH]1"
    pt_smis = protonate_mol(smi, ph=7.0, tph=1.5)
    print(pt_smis)


   


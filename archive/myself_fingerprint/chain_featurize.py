import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops
import pandas as pd

# --------------------------------------------------
# 1. 定義 59 維 chain-level 特徵鍵
# --------------------------------------------------
chain_keys = [
    # 一、QSPR 描述符 (29 項)
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumAtoms', 'NumValenceElectrons',
    'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'FractionCSP3', 'TPSA',
    'MolLogP', 'MolMR', 'HallKierAlpha', 'BalabanJ', 'BertzCT', 'Kappa1', 'Kappa2',
    'Kappa3', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
    'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    # 二、形態學/拓撲指標 (30 項)
    'NumRings', 'NumRingSystems', 'MaxRingSize', 'MinRingSize', 'AverageRingSize',
    'NumSaturatedCarbocycles', 'NumAromaticRings', 'NumAliphaticRings', 'NumHeterocycles',
    'NumSpiroAtoms', 'NumBridgeheadAtoms', 'NumStereoCenters', 'NumRadicalElectrons',
    'GraphDiameter', 'GraphRadius', 'EccentricConnectivityIndex', 'VertexConnectivity',
    'WienerIndex', 'ZagrebIndex', 'LongestPath', 'NumHeavyAtoms',
    'LongestSideChain', 'ShortestSideChain', 'AverageSideChainLength',
    'LongestSideChain_noRing', 'ShortestSideChain_noRing', 'AverageSideChainLength_noRing',
    'ShortestInterRingDist', 'LongestInterRingDist', 'AverageInterRingDist'
]

# --------------------------------------------------
# 2. 定義抽取函式
# --------------------------------------------------
from rdkit.Chem import rdmolfiles

def extract_chain_features(mol):
    feats = {}
    # QSPR descriptors
    feats['MolWt'] = Descriptors.MolWt(mol)
    feats['ExactMolWt'] = Descriptors.ExactMolWt(mol)
    feats['HeavyAtomCount'] = mol.GetNumHeavyAtoms()
    feats['NumAtoms'] = mol.GetNumAtoms()
    feats['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    feats['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    feats['NumHDonors'] = Descriptors.NumHDonors(mol)
    feats['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    feats['FractionCSP3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    feats['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
    feats['MolLogP'] = Descriptors.MolLogP(mol)
    feats['MolMR'] = Descriptors.MolMR(mol)
    feats['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
    feats['BalabanJ'] = Descriptors.BalabanJ(mol)
    feats['BertzCT'] = Descriptors.BertzCT(mol)
    feats['Kappa1'] = Descriptors.Kappa1(mol)
    feats['Kappa2'] = Descriptors.Kappa2(mol)
    feats['Kappa3'] = Descriptors.Kappa3(mol)
    feats['Chi0'] = Descriptors.Chi0(mol)
    feats['Chi0n'] = Descriptors.Chi0n(mol)
    feats['Chi0v'] = Descriptors.Chi0v(mol)
    feats['Chi1'] = Descriptors.Chi1(mol)
    feats['Chi1n'] = Descriptors.Chi1n(mol)
    feats['Chi1v'] = Descriptors.Chi1v(mol)
    feats['Chi2n'] = Descriptors.Chi2n(mol)
    feats['Chi2v'] = Descriptors.Chi2v(mol)
    feats['Chi3n'] = Descriptors.Chi3n(mol)
    feats['Chi3v'] = Descriptors.Chi3v(mol)
    feats['Chi4n'] = Descriptors.Chi4n(mol)
    feats['Chi4v'] = Descriptors.Chi4v(mol)

    # Morphological / topology features
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    feats['NumRings'] = ring_info.NumRings()

    # NumRingSystems via union-find
    parent = list(range(len(atom_rings)))
    def find(i):
        if parent[i]!=i: parent[i]=find(parent[i])
        return parent[i]
    def union(i,j):
        pi,pj = find(i),find(j)
        if pi!=pj: parent[pi]=pj
    for i in range(len(atom_rings)):
        for j in range(i+1, len(atom_rings)):
            if set(atom_rings[i]) & set(atom_rings[j]): union(i,j)
    feats['NumRingSystems'] = len({find(i) for i in range(len(atom_rings))})

    sizes = [len(r) for r in atom_rings] or [0]
    feats['MaxRingSize'] = max(sizes)
    feats['MinRingSize'] = min(sizes)
    feats['AverageRingSize'] = float(sum(sizes)/len(sizes))

    # Saturated carbocycles: rings all C and single, non-aromatic bonds
    sat=0
    arom=0
    hetero=0
    for r in atom_rings:
        atoms = [mol.GetAtomWithIdx(i) for i in r]
        bonds = []
        ok_sat=True
        for idx in range(len(r)):
            a1,a2 = r[idx], r[(idx+1)%len(r)]
            b=mol.GetBondBetweenAtoms(a1,a2)
            if b.GetIsAromatic() or b.GetBondType().name!='SINGLE': ok_sat=False
        if ok_sat and all(a.GetSymbol()=='C' for a in atoms): sat+=1
        # aromatic ring if any bond aromatic
        if any(mol.GetBondBetweenAtoms(r[i],r[(i+1)%len(r)]).GetIsAromatic() for i in range(len(r))): arom+=1
        # heterocycle if any atom not C
        if any(a.GetSymbol()!='C' for a in atoms): hetero+=1
    feats['NumSaturatedCarbocycles'] = sat
    feats['NumAromaticRings'] = arom
    feats['NumAliphaticRings'] = feats['NumRings'] - arom
    feats['NumHeterocycles'] = hetero

    # Spiro and bridgehead atoms
    ring_membership = {a.GetIdx(): ring_info.NumAtomRings(a.GetIdx()) for a in mol.GetAtoms()}
    feats['NumSpiroAtoms'] = sum(1 for v in ring_membership.values() if v==2)
    feats['NumBridgeheadAtoms'] = sum(1 for v in ring_membership.values() if v>2)

    # Stereo centers
    feats['NumStereoCenters'] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    feats['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)

    # Graph metrics
    dmat = rdmolops.GetDistanceMatrix(mol)
    feats['GraphDiameter'] = float(np.max(dmat))
    feats['GraphRadius'] = float(min(np.max(dmat,axis=1)))
    feats['EccentricConnectivityIndex'] = Descriptors.EccentricConnectivityIndex(mol)
    feats['VertexConnectivity'] = Descriptors.VertexConnectivity(mol)
    feats['WienerIndex'] = Descriptors.WienerIndex(mol)
    feats['ZagrebIndex'] = Descriptors.ZagrebIndex(mol)
    feats['LongestPath'] = feats['GraphDiameter']
    feats['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()

    # Side chain distances
    ring_atoms_set = set([i for r in atom_rings for i in r])
    nonring = [i for i in range(mol.GetNumAtoms()) if i not in ring_atoms_set]
    if ring_atoms and nonring:
        side_dists = [min(dmat[i,j] for j in ring_atoms) for i in nonring]
        feats['LongestSideChain'] = float(max(side_dists))
        feats['ShortestSideChain'] = float(min(side_dists))
        feats['AverageSideChainLength'] = float(sum(side_dists)/len(side_dists))
        feats['LongestSideChain_noRing'] = feats['LongestSideChain']
        feats['ShortestSideChain_noRing'] = feats['ShortestSideChain']
        feats['AverageSideChainLength_noRing'] = feats['AverageSideChainLength']
    else:
        for k in ['LongestSideChain','ShortestSideChain','AverageSideChainLength',
                  'LongestSideChain_noRing','ShortestSideChain_noRing','AverageSideChainLength_noRing']:
            feats[k] = 0.0

    # Inter-ring distances
    if len(ring_atoms)>1:
        inter = [dmat[i,j] for i in ring_atoms for j in ring_atoms if i<j]
        feats['ShortestInterRingDist'] = float(min(inter))
        feats['LongestInterRingDist'] = float(max(inter))
        feats['AverageInterRingDist'] = float(sum(inter)/len(inter))
    else:
        feats['ShortestInterRingDist'] = feats['LongestInterRingDist'] = feats['AverageInterRingDist'] = 0.0

    # 返回固定順序列表
    return [feats[k] for k in chain_keys]

# --------------------------------------------------
# 3. 主程式示例
# --------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    matrix = []
    for smi in df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        matrix.append(extract_chain_features(mol) if mol else [0.0]*len(chain_keys))
    X_chain = np.array(matrix)
    print("Chain-level 特徵矩陣形狀：", X_chain.shape)
    print("前 5 個特徵名稱：", chain_keys[:5])

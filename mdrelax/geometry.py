"""Selection of relaxation-active bond vectors from an MDAnalysis Universe.

* Backbone amide N-H pairs (excluding proline and the N-terminus).
* Side-chain methyl groups (ALA/VAL/LEU/ILE/THR/MET): the methyl carbon, its
  three protons, and the adjacent "axis" carbon defining the C-C symmetry axis.

Methyl labels follow the ABSURDer convention ``RESNAME<resid>-<Cname><Hbase>``
(e.g. ``ALA41-CBHB``, ``VAL71-CG1HG1``) so results can be matched by label.
"""

import numpy as np

# residue -> list of (methyl_carbon, [candidate H names], axis_carbon)
METHYL_DEFS = {
    'ALA': [('CB',  ['HB1', 'HB2', 'HB3', '1HB', '2HB', '3HB'], 'CA')],
    'VAL': [('CG1', ['HG11', 'HG12', 'HG13', '1HG1', '2HG1', '3HG1'], 'CB'),
            ('CG2', ['HG21', 'HG22', 'HG23', '1HG2', '2HG2', '3HG2'], 'CB')],
    'THR': [('CG2', ['HG21', 'HG22', 'HG23', '1HG2', '2HG2', '3HG2'], 'CB')],
    'ILE': [('CG2', ['HG21', 'HG22', 'HG23', '1HG2', '2HG2', '3HG2'], 'CB'),
            ('CD1', ['HD11', 'HD12', 'HD13', '1HD1', '2HD1', '3HD1', 'HD1', 'HD2', 'HD3'], 'CG1')],
    'LEU': [('CD1', ['HD11', 'HD12', 'HD13', '1HD1', '2HD1', '3HD1'], 'CG'),
            ('CD2', ['HD21', 'HD22', 'HD23', '1HD2', '2HD2', '3HD2'], 'CG')],
    'MET': [('CE',  ['HE1', 'HE2', 'HE3', '1HE', '2HE', '3HE'], 'SD')],
}


def find_nh_pairs(universe):
    """Return backbone amide N-H pairs (excludes PRO and residue 1).

    Returns
    -------
    list of dict with keys: resid, resname, N (Atom), H (Atom)
    """
    universe.trajectory[0]
    pairs = []
    backbone_N = universe.select_atoms("protein and backbone and name N")
    if len(backbone_N) == 0:
        return pairs
    n_terminus = int(min(a.resid for a in backbone_N))
    for n_atom in backbone_N:
        if n_atom.resname == "PRO":
            continue
        if n_atom.resid == n_terminus:      # skip N-terminal NH3+
            continue
        h_sel = universe.select_atoms(
            f"resid {n_atom.resid} and name H HN H1 HT1 1H")
        if len(h_sel) == 0:
            h_sel = universe.select_atoms(
                f"name H* and around 1.2 (index {n_atom.index})")
        if len(h_sel) == 0:
            continue
        dists = np.linalg.norm(h_sel.positions - n_atom.position, axis=1)
        h_atom = h_sel[int(np.argmin(dists))]
        pairs.append(dict(resid=n_atom.resid, resname=n_atom.resname,
                          N=n_atom, H=h_atom))
    return pairs


def _first_named(residue, names):
    for nm in names:
        sel = residue.atoms.select_atoms(f"name {nm}")
        if len(sel) > 0:
            return sel[0]
    return None


def find_methyl_groups(universe):
    """Return methyl groups for ALA/VAL/LEU/ILE/THR/MET.

    Returns
    -------
    list of dict with keys:
        resid, resname, methyl_name, label,
        C (methyl carbon Atom), H (list of 3 Atoms), C_axis (axis carbon Atom)
    """
    universe.trajectory[0]
    groups = []
    for res in universe.residues:
        if res.resname not in METHYL_DEFS:
            continue
        for (c_name, h_names, axis_name) in METHYL_DEFS[res.resname]:
            c_atom = _first_named(res, [c_name])
            if c_atom is None:
                continue
            c_axis = _first_named(res, [axis_name])
            if c_axis is None:
                heavy = res.atoms.select_atoms("not name H*")
                d = np.linalg.norm(heavy.positions - c_atom.position, axis=1)
                d[heavy.indices == c_atom.index] = 1e9
                c_axis = heavy[int(np.argmin(d))]

            h_atoms = []
            for hn in h_names:
                sel = res.atoms.select_atoms(f"name {hn}")
                if len(sel) > 0 and sel[0] not in h_atoms:
                    h_atoms.append(sel[0])
                if len(h_atoms) == 3:
                    break
            if len(h_atoms) < 3:
                near = universe.select_atoms(
                    f"name H* and around 1.2 (index {c_atom.index})")
                for ha in near:
                    if ha not in h_atoms:
                        h_atoms.append(ha)
                    if len(h_atoms) == 3:
                        break
            if len(h_atoms) < 3:
                continue

            h_base = h_atoms[0].name[:-1]  # HB1 -> HB, HG11 -> HG1
            label = f"{res.resname}{res.resid}-{c_name}{h_base}"
            groups.append(dict(resid=res.resid, resname=res.resname,
                               methyl_name=c_name, label=label,
                               C=c_atom, H=h_atoms[:3], C_axis=c_axis))
    return groups


def collect_vectors(universe, pairs, start=None, stop=None, step=None):
    """Collect bond vectors (H - N) for NH pairs over the trajectory.

    Returns ndarray of shape (n_frames, n_pairs, 3).
    """
    frames = universe.trajectory[start:stop:step]
    n_frames = len(frames)
    n = len(pairs)
    vecs = np.zeros((n_frames, n, 3), dtype=np.float32)
    n_idx = np.array([p['N'].index for p in pairs])
    h_idx = np.array([p['H'].index for p in pairs])
    for i, _ in enumerate(universe.trajectory[start:stop:step]):
        pos = universe.atoms.positions
        vecs[i] = pos[h_idx] - pos[n_idx]
    return vecs


def collect_methyl_ch_vectors(universe, groups, start=None, stop=None, step=None):
    """Collect the three C-H vectors per methyl over the trajectory.

    Returns ndarray of shape (n_frames, n_methyls, 3, 3) -> last two axes are
    (which_H, xyz).
    """
    n_frames = len(universe.trajectory[start:stop:step])
    n = len(groups)
    c_idx = np.array([g['C'].index for g in groups])
    h_idx = np.array([[h.index for h in g['H']] for g in groups])  # (n, 3)
    vecs = np.zeros((n_frames, n, 3, 3), dtype=np.float32)
    for i, _ in enumerate(universe.trajectory[start:stop:step]):
        pos = universe.atoms.positions
        c = pos[c_idx]                     # (n, 3)
        for hj in range(3):
            vecs[i, :, hj, :] = pos[h_idx[:, hj]] - c
    return vecs

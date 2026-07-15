# Ubiquitin reference case for pdbinertia / quadric_diffusion

Verbatim copies of the ubiquitin test case distributed with Arthur Palmer's
`pdbinertia` and `quadric_diffusion` programs. `mdrelax.tumbling` reimplements
both in pure Python, and `tests/test_quadric_reference.py` checks it against the
outputs here. Vendored so the reference tests always run.

| file | origin | role |
|---|---|---|
| `1ubq.prot.pdb` | both | ubiquitin crystal structure (PDB 1UBQ, protonated); the only structural input |
| `ubq.inertia.out` | pdbinertia | expected: centre of mass, principal moments, rotation matrix |
| `ubq.tm.input` | quadric | input: per-residue `resid  tau_M(ns)  dtau_M` for 55 residues |
| `ubq.output` | quadric | expected: iso / axial / aniso diffusion tensors, chi-square, F-statistics |

Two details that matter when comparing against `ubq.output`:

* Its `Input Data` table lists `D/10^7` and `dD/10^7`, derived from `ubq.tm.input`
  as `D = 1/(6 tau_M)` and `dD = D dtau_M / tau_M`.
* The `x y z` columns of that table are **H->N** unit vectors, whereas `mdrelax`
  builds N->H. P2 is even, so only the axis matters.

quadric was run on `1ubq.trans.pdb` (the same structure translated to its centre
of mass by pdbinertia) rather than `1ubq.prot.pdb`. Bond vectors are unaffected
by translation, so the single PDB here reproduces both programs.

Regenerate with:

```bash
pdbinertia -r 1ubq.prot.pdb 1ubq.prot.inertia.pdb  > ubq.inertia.out
quadric_diffusion ubq.in                           > ubq.output
```

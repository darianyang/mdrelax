# mdrelax

Nuclear spin-relaxation rates from MD simulation trajectories.

`mdrelax` computes, directly from an MD trajectory:

* **Backbone amide ¹⁵N–¹H**: R₁, R₂, and the heteronuclear ¹⁵N-{¹H} NOE.
* **Side-chain methyl ²H**: the three experimentally measured rates R(D_z),
  R(D_y), and R(3D_z²−2) (quadrupolar order).

It is a pure-Python (numpy / scipy / MDAnalysis) reimplementation of the standard
model-free (backbone) and spectral-density-mapping (methyl) workflows. The methyl
path reproduces the [ABSURDer](https://github.com/felixkum/ABSURDer) results
**without** GROMACS, pdbinertia, or quadric_diffusion: the per-methyl tumbling
time τ_R is obtained from a pure-Python rotational diffusion-tensor fit, which
reproduces pdbinertia and quadric_diffusion on their own ubiquitin test case
(see [Validation](#validation)).

## Install

```bash
pip install -e .            # runtime deps: numpy, scipy, pandas, MDAnalysis
pip install -e ".[test,plot]"   # + pytest, matplotlib
```

## Usage

### Python API

```python
from mdrelax import NHRelaxation, MethylRelaxation

# Backbone NH at 600 MHz (tau_c estimated from the trajectory if not given)
df_nh = NHRelaxation("topol.pdb", "traj.xtc", fields_MHz=600.0, tau_c_ns=10.5).run()

# Side-chain methyl 2H at 950 MHz.  `trajectory` should retain overall tumbling
# (used for tau_M); `trajectory_fitted` has tumbling removed (methyl internal
# motion).  If `trajectory_fitted` is omitted, the trajectory is CA-aligned.
df_me = MethylRelaxation("topol.pdb", "traj_nopbc.xtc",
                         trajectory_fitted="traj_rot_trans.xtc",
                         field_MHz=950.0).run()
```

Both `run()` calls return a `pandas.DataFrame` (per residue / per methyl).

### Choosing the diffusion model

The rotational-diffusion model fit to the backbone τ_M depends on the protein's
shape: `iso` (1 parameter) suits a globular one, `axial` (4 parameters; what
ABSURDer assumes) a prolate/oblate one, `aniso` (6 parameters) one where no two
principal values are alike.

By default `MethylRelaxation` does not assume: it fits **all three and picks
between them with an F-test**, the criterion quadric's output is built around.
That costs ~0.5 s against the minutes the correlation functions take, so it is
free in practice. Pass `diffusion_model="axial"` (or `"iso"`/`"aniso"`) to force
one — e.g. for ABSURDer parity — and it is ignored entirely if you supply
`tau_R_ns` yourself. The fit used is left on `.diffusion`, the three candidates
on `.diffusion_trials`.

The F-test assumes independent normal residuals; per-residue τ_M are neither, so
treat the choice as a good default rather than the last word. To drive it
yourself:

```python
from mdrelax import tumbling

fit, trials = tumbling.select_diffusion_model(tauM_ps, nh_vectors)
print(tumbling.describe(fit))            # e.g. "axial: Diso=4.013e7 s^-1 ..."
print(fit["selection"])                  # [(simpler, complex, F, p, accepted), ...]
```

### Command line

```bash
mdrelax-nh      topol.pdb traj.xtc --field 500 600 800 --tau_c 10.5 -o nh.csv
mdrelax-methyl  topol.pdb traj_nopbc.xtc --fitted traj_rot_trans.xtc -f 950 -o ch3.csv

# force a model instead of the default F-test selection
mdrelax-methyl  topol.pdb traj_nopbc.xtc --fitted traj_rot_trans.xtc \
                --diffusion-model axial -o ch3.csv
```

## How it works

**Backbone NH** (`mdrelax.nh`): align the trajectory to Cα (removes tumbling) →
per-residue N–H P₂ autocorrelation → Lipari–Szabo / Extended Model-Free fit →
spectral density J(ω) with the overall τ_c reintroduced → dipolar + CSA rates.

**Methyl ²H** (`mdrelax.methyl`, ABSURDer method):
1. C–H P₂ autocorrelation on the tumbling-removed trajectory, averaged over the
   three methyl protons → internal C(t).
2. Long-time plateau S² (tail average) + a 6-exponential internal fit.
3. Per-methyl τ_R from backbone τ_M → rotational diffusion tensor
   (`mdrelax.tumbling`, replacing pdbinertia + quadric).
4. Multi-exponential J(ω) with τ_R reintroduced → the three ²H quadrupolar rates
   R(D_z), R(D_y), R(3D_z²−2).

## Package layout

```
mdrelax/
  constants.py         physical constants, gyromagnetic ratios, coupling prefactors
  acf.py               FFT P2 autocorrelation (+ direct reference)
  geometry.py          NH-pair / methyl-group selection from MDAnalysis
  fitting.py           LS / EMF (NH) and 6-exponential internal (methyl) fits
  spectral_density.py  J(omega): Lipari-Szabo, EMF, anisotropic, multi-exponential
  rates.py             NH (dipolar+CSA) and methyl 2H (quadrupolar) rate expressions
  tumbling.py          tau_c, backbone tau_M, diffusion tensor, per-methyl tau_R
  nh.py                NHRelaxation orchestrator
  methyl.py            MethylRelaxation orchestrator
  cli.py               mdrelax-nh / mdrelax-methyl entry points
reference/absurder/    original ABSURDer scripts (provenance / cross-check)
examples/              validate_nh.py, validate_ch3.py, validate_ch3_exp.py
tests/                 pytest suite
  data/ubq/            pdbinertia + quadric ubiquitin test case (their outputs)
```

## Validation

Run the suite:

```bash
pytest -q
```

* **Tumbling vs pdbinertia + quadric_diffusion**, on the ubiquitin test case that
  ships with those two programs — `tests/test_quadric_reference.py`. Everything
  is driven from their single `1ubq.prot.pdb` plus the per-residue τ_M in
  `ubq.tm.input` (both vendored under `tests/data/ubq/`, so it always runs), and
  checked against their own published output:

  | reference | quantity | theirs | ours |
  |---|---|---|---|
  | pdbinertia | principal moments | 934104 / 844541 / 594139 | within 1e-4 rel. |
  | pdbinertia | centre of mass | 30.3128 28.8001 15.3504 | within 5e-4 Å |
  | quadric | Diso (iso), χ²_red | 4.05073e7, 12.9239 | 4.05073e7, 12.9239 |
  | quadric | Diso, Dpar/Dper (axial) | 4.01315e7, 1.15315 | 4.01316e7, 1.15315 |
  | quadric | Diso, Dxx:Dyy:Dzz (aniso) | 4.01763e7, 3.769:3.863:4.421 | same to 1e-3 |
  | quadric | F(iso→axial), F(axial→aniso) | 12.101, 0.4746 | 12.101, 0.4746 |

  The remaining pdbinertia difference is an atomic-mass-table digit (total mass
  8563.85 vs 8563.90), not a difference in method. Unlike quadric we fit the
  tensor orientation, so the structure needs no pdbinertia pre-alignment — a
  test rotates the input frame and checks the tensor co-rotates. Those F values
  are also what the default model selection consumes: it lands on `axial` for
  ubiquitin, its accepted description.
* **Backbone NH** on `data-md-t4l` (10 ns T4L) vs experimental 600 MHz data
  (`data-nh`): mean R₁ 1.13 vs 1.18, R₂ 13.1 vs 13.9, NOE 0.86 vs 0.79 s⁻¹.
  Per-residue detail is limited by the short trajectory; magnitudes are correct.
  → `python examples/validate_nh.py`
* **Methyl ²H** — two cross-checks with deliberately different scope:
  * **vs ABSURDer** `rates.pkl` (1 µs ff15ipq T4L) — `examples/validate_ch3.py`.
    Consumes ABSURDer's *precomputed* TCFs so that only the fit / J(ω) / rate
    port is under test; it therefore pins ABSURDer's own run settings
    (`accuracy`, `ct_lim`, `wD`) to compare like for like. Reproduces all three
    rates at **r > 0.9999, MAD < 0.2 s⁻¹**. The pure-Python τ_R matches the
    pdbinertia+quadric reference to ~0.15 ns (Diso within ~1.3 %).
  * **vs experiment** (`data-ch3/experimental/`, 73 measured methyls @ 950 MHz)
    — `examples/validate_ch3_exp.py`. The real workflow: hand it a trajectory
    and `MethylRelaxation` computes the TCFs itself, deriving the whole time
    axis from the trajectory. Accepts `--topology/--traj/--fitted/--field`, so
    it runs on your own data.

    Note τ_R: overall tumbling (τ_c ≈ 13 ns for T4L) can only be measured from a
    trajectory much longer than τ_c. The bundled 10 ns trajectory is ABSURDer's
    *methyl* block length — right for fast internal motion, far too short for
    tumbling (it gives τ_c ≈ 8.7 ns vs the true ≈ 12.9 ns). So τ_R defaults to
    the 1 µs backbone fit, exactly as ABSURDer does (`--lblocks_m 10000` for
    methyls, `--lblocks_bb 1000000` for the backbone). Use `--estimate-tau-R`
    when your trajectory is long compared with τ_c.

Reference data lives in `data-nh/`, `data-ch3/` (experimental NMR + ABSURDer
reweighting data), and `data-ch3-ff15ipq/` (ABSURDer TCFs, τ_R, and rates for the
1 µs ff15ipq T4L trajectory).

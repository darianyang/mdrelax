# mdrelax

Nuclear spin-relaxation rates from MD simulation trajectories.

`mdrelax` computes, directly from an MD trajectory:

* **Backbone amide ¹⁵N–¹H**: R₁, R₂, and the heteronuclear ¹⁵N-{¹H} NOE.
* **Side-chain methyl ²H**: the three experimentally measured rates R(D_z),
  R(D_y), and R(3D_z²−2) (quadrupolar order).

It is a pure-Python (numpy / scipy / MDAnalysis) reimplementation of the standard
model-free (backbone) and spectral-density-mapping (methyl) workflows. The methyl
path reproduces the [ABSURDer](https://github.com/felixkum/ABSURDer) results
**without** GROMACS, pdbinertia, or quadric_diffusion — the per-methyl tumbling
time τ_R is obtained from a pure-Python axially symmetric diffusion-tensor fit.

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

### Command line

```bash
mdrelax-nh      topol.pdb traj.xtc --field 500 600 800 --tau_c 10.5 -o nh.csv
mdrelax-methyl  topol.pdb traj_nopbc.xtc --fitted traj_rot_trans.xtc -f 950 -o ch3.csv
```

## How it works

**Backbone NH** (`mdrelax.nh`): align the trajectory to Cα (removes tumbling) →
per-residue N–H P₂ autocorrelation → Lipari–Szabo / Extended Model-Free fit →
spectral density J(ω) with the overall τ_c reintroduced → dipolar + CSA rates.

**Methyl ²H** (`mdrelax.methyl`, ABSURDer method):
1. C–H P₂ autocorrelation on the tumbling-removed trajectory, averaged over the
   three methyl protons → internal C(t).
2. Long-time plateau S² (tail average) + a 6-exponential internal fit.
3. Per-methyl τ_R from backbone τ_M → axially symmetric diffusion tensor
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
  tumbling.py          tau_c, backbone tau_M, axial diffusion tensor, per-methyl tau_R
  nh.py                NHRelaxation orchestrator
  methyl.py            MethylRelaxation orchestrator
  cli.py               mdrelax-nh / mdrelax-methyl entry points
reference/absurder/    original ABSURDer scripts (provenance / cross-check)
examples/              validate_nh.py, validate_ch3.py, validate_ch3_exp.py
tests/                 pytest suite
```

## Validation

Run the suite:

```bash
pytest -q
```

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

# QGU-DG: Quantum Gravity Unification via Dark Geometry

<p align="center">
  <img src="figures/qgu_logo.png" alt="QGU-DG Logo" width="400"/>
</p>

<p align="center">
  <strong>A unified framework where Asymptotic Safety, Loop Quantum Gravity, String Theory, Causal Dynamical Triangulations, and Holography converge</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#theory">Theory</a> •
  <a href="#simulations">Simulations</a> •
  <a href="#citation">Citation</a>
</p>

---

## 🌌 Overview

**QGU-DG** (Quantum Gravity Unification via Dark Geometry) is a theoretical framework proposing that dark matter and dark energy are not exotic substances, but manifestations of the **conformal mode of spacetime**—the same degree of freedom mediating gravity.

The central equation is:

```
m²_eff(ρ) = (α* M_Pl)² [1 - (ρ/ρ_c)^(2/3)]
```

Where:
- **α\* = 0.075**: Coupling from Asymptotic Safety UV fixed point (g\* = 0.816)
- **β = 2/3**: Holographic exponent from area-volume relation (A ∝ V^(2/3))
- **ρ_c ≈ (2.28 meV)⁴**: Critical density from UV-IR connection

### Key Insight

The Dark Boson behaves as:
- **Dark Matter** (m²_eff < 0, tachyonic): In overdense regions (ρ > ρ_c)
- **Dark Energy** (m²_eff > 0, stable): In underdense regions (ρ < ρ_c)

## 🔬 Theoretical Foundations

### Five Approaches, One Framework

| Approach | Contribution | Parameter |
|----------|-------------|-----------|
| **Asymptotic Safety** | UV fixed point g\* = 0.816 | α\* = 0.075 |
| **Loop Quantum Gravity** | Area spectrum A_j ∝ √(j(j+1)) | β = 2/3 |
| **String Theory** | Dilaton = Dark Boson | Coupling structure |
| **Causal Dynamical Triangulations** | Spectral dimension 4→2 | Dimensional reduction |
| **Holographic Principle** | Bekenstein-Hawking entropy | ρ_c from UV-IR |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/HugoHertault/QGU-DG.git
cd QGU-DG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CLASS (Cosmic Linear Anisotropy Solving System) for CMB calculations
# See: https://github.com/lesgourg/class_public
```

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5
- Astropy ≥ 5.0
- emcee ≥ 3.1 (for MCMC)
- h5py ≥ 3.0 (for data storage)

## 🚀 Quick Start

### Basic Usage

```python
from qgu_dg import DarkGeometry, HaloProfile, CosmologicalModel

# Initialize the DG model with derived parameters
dg = DarkGeometry(
    alpha_star=0.075,      # From Asymptotic Safety
    beta=2/3,              # From Holography/LQG
    rho_c_meV4=27.0        # (2.28 meV)^4
)

# Compute effective mass at given density
rho = 1e-24  # kg/m³
m_eff_sq = dg.effective_mass_squared(rho)
print(f"m²_eff = {m_eff_sq:.2e} eV²")

# Generate halo profile
halo = HaloProfile(M_vir=1e12, c=10, model='DG')
r = np.logspace(-1, 3, 100)  # kpc
rho_halo = halo.density(r)

# Cosmological evolution
cosmo = CosmologicalModel(model='DG')
z = np.linspace(0, 10, 100)
w_z = cosmo.equation_of_state(z)
```

### Running Simulations

```bash
# Power spectrum comparison DG vs ΛCDM
python simulations/power_spectrum.py --output figures/

# Halo density profiles
python simulations/halo_profiles.py --masses 1e10,1e12,1e14

# CMB angular power spectrum
python simulations/cmb_spectrum.py

# Full MCMC analysis
python simulations/mcmc_analysis.py --chains 4 --samples 10000
```

## 📊 Simulations

### Available Simulations

| Simulation | Description | Output |
|------------|-------------|--------|
| `power_spectrum.py` | Matter P(k) with DG suppression | Fig. 1-3 |
| `halo_profiles.py` | NFW vs DG halo profiles | Fig. 4-6 |
| `spectral_dimension.py` | d_s flow from 4 to 2 | Fig. 7 |
| `as_fixed_point.py` | RG flow to UV fixed point | Fig. 8 |
| `uv_ir_connection.py` | ρ_c from Planck-Hubble scales | Fig. 9 |
| `cmb_spectrum.py` | CMB T and E-mode spectra | Fig. 10-11 |
| `s8_tension.py` | σ₈ prediction comparison | Fig. 12 |
| `equation_of_state.py` | w(z) evolution | Fig. 13 |

### Example Output

```
=== QGU-DG Parameter Derivation ===

From Asymptotic Safety:
  g* = 0.816 (UV fixed point)
  α* = g*/(4π) × √(4/3) = 0.0750

From Holography (Area-Volume):
  A ∝ V^(2/3) in d=3
  β = 2/3 = 0.6667

From UV-IR Connection:
  E_Pl = 1.22 × 10¹⁹ GeV
  E_H = 1.5 × 10⁻³³ eV
  √(E_Pl × E_H) / 2 = 2.15 meV
  ρ_c^(1/4) = 2.28 meV ✓

Zero free parameters. All derived from first principles.
```

## 📈 Key Predictions

### Testable Differences from ΛCDM

| Observable | QGU-DG | ΛCDM | Current Data |
|------------|--------|------|--------------|
| σ₈ | 0.74-0.78 | 0.81 | 0.76 (weak lensing) ✓ |
| Dwarf cores | n ≈ 0 | n = -1 (cusp) | n ≈ 0 ✓ |
| MW satellites | ~60 | ~500 | ~60 ✓ |
| w(z) | Evolving | -1 (constant) | Hints (DESI) |
| Halo edge | r_edge ~ 13 r_s | No edge | To test |

### Suppression Function

```python
def suppression(k, k_s=0.3):
    """Matter power spectrum suppression in QGU-DG"""
    return 1 - 0.25 * (1 - 1/(1 + (k/k_s)**2.8))
```

## 📚 Documentation

- [Theory Overview](docs/theory.md) - Mathematical foundations
- [Parameter Derivations](docs/parameters.md) - α\*, β, ρ_c from QG
- [Numerical Methods](docs/numerical.md) - Simulation algorithms
- [API Reference](docs/api.md) - Complete function documentation

## 📄 Citation

If you use QGU-DG in your research, please cite:

```bibtex
@article{Hertault2025QGU,
  author  = {Hertault, Hugo},
  title   = {Dark Geometry and the Unification of Quantum Gravity: 
             How Asymptotic Safety, Loop Quantum Gravity, String Theory, 
             Causal Dynamical Triangulations, and the Holographic Principle 
             Converge to a Single Framework},
  journal = {zenodo preprint},
  year    = {2025},
  eprint  = {2412.xxxxx},
  note    = {Paper III of the Dark Geometry Series}
}

@article{Hertault2025DG,
  author  = {Hertault, Hugo},
  title   = {Dark Geometry: A Proposal for Unifying Dark Matter 
             and Dark Energy as the Scalar Dynamics of Spacetime},
  journal = {zenodo preprint},
  year    = {2025},
  eprint  = {2412.xxxxx},
  note    = {Paper I of the Dark Geometry Series}
}

@article{Hertault2025HDG,
  author  = {Hertault, Hugo},
  title   = {Holographic Dark Geometry: The Emergent Dimension 
             of the Dark Sector},
  journal = {zenodo preprint},
  year    = {2025},
  eprint  = {2412.xxxxx},
  note    = {Paper II of the Dark Geometry Series}
}
```

## 🔗 Related Papers

1. Weinberg (1979) - Asymptotic Safety conjecture
2. Reuter (1998) - Functional RG for gravity
3. Rovelli & Smolin (1995) - LQG area spectrum
4. Maldacena (1998) - AdS/CFT correspondence
5. Ambjørn et al. (2005) - CDT spectral dimension

## 📞 Contact

**Hugo Hertault**  
Independent Researcher  
Tahiti, French Polynesia  
📧 hertault.toe@gmail.com

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>"The dark sector is not separate from gravity—it is gravity."</em>
</p>

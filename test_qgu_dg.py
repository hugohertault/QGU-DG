"""
Unit tests for QGU-DG module
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from qgu_dg import (
    DarkGeometry, 
    QGUParameters, 
    AsymptoticSafety, 
    LoopQuantumGravity,
    HolographicPrinciple,
    SpectralDimension
)


class TestQGUParameters:
    """Test parameter derivations."""
    
    def test_alpha_from_gstar(self):
        """Verify α* = g*/(4π) × √(4/3)."""
        params = QGUParameters()
        alpha_calc = (params.g_star / (4 * np.pi)) * np.sqrt(4/3)
        assert np.isclose(alpha_calc, params.alpha_star, rtol=0.01)
    
    def test_beta_value(self):
        """Verify β = 2/3."""
        params = QGUParameters()
        assert np.isclose(params.beta, 2/3, rtol=1e-10)
    
    def test_beta_from_geometry(self):
        """Verify β = (d-1)/d for d=3."""
        d = 3
        beta_calc = (d - 1) / d
        assert np.isclose(beta_calc, 2/3)


class TestDarkGeometry:
    """Test Dark Geometry core class."""
    
    def test_effective_mass_at_critical(self):
        """m²_eff should be zero at ρ = ρ_c."""
        dg = DarkGeometry()
        m2 = dg.effective_mass_squared(dg.rho_c)
        assert np.isclose(m2, 0, atol=1e-10 * dg.m_scale_sq)
    
    def test_dm_regime(self):
        """ρ > ρ_c should give m² < 0 (dark matter)."""
        dg = DarkGeometry()
        rho_high = 10 * dg.rho_c
        m2 = dg.effective_mass_squared(rho_high)
        assert m2 < 0
        assert dg.is_dark_matter_regime(rho_high)
    
    def test_de_regime(self):
        """ρ < ρ_c should give m² > 0 (dark energy)."""
        dg = DarkGeometry()
        rho_low = 0.1 * dg.rho_c
        m2 = dg.effective_mass_squared(rho_low)
        assert m2 > 0
        assert dg.is_dark_energy_regime(rho_low)
    
    def test_array_input(self):
        """Should handle array inputs."""
        dg = DarkGeometry()
        rho_array = np.logspace(-3, 3, 100) * dg.rho_c
        m2_array = dg.effective_mass_squared(rho_array)
        assert len(m2_array) == 100


class TestAsymptoticSafety:
    """Test Asymptotic Safety calculations."""
    
    def test_fixed_point_values(self):
        """Check fixed point is at expected values."""
        AS = AsymptoticSafety()
        assert np.isclose(AS.g_star, 0.816, rtol=0.01)
        assert np.isclose(AS.lambda_star, 0.193, rtol=0.05)
    
    def test_alpha_derivation(self):
        """Verify α* derivation from g*."""
        AS = AsymptoticSafety()
        alpha = AS.alpha_from_g_star()
        assert np.isclose(alpha, 0.075, rtol=0.01)


class TestLoopQuantumGravity:
    """Test LQG calculations."""
    
    def test_area_spectrum(self):
        """Area spectrum should scale as √(j(j+1))."""
        lqg = LoopQuantumGravity()
        
        j1, j2 = 10, 20
        A1 = lqg.area_eigenvalue(j1)
        A2 = lqg.area_eigenvalue(j2)
        
        ratio_expected = np.sqrt(j2*(j2+1)) / np.sqrt(j1*(j1+1))
        ratio_actual = A2 / A1
        
        assert np.isclose(ratio_actual, ratio_expected, rtol=1e-5)
    
    def test_beta_verification(self):
        """Verify A ∝ V^(2/3) from LQG spectra."""
        lqg = LoopQuantumGravity()
        results = lqg.verify_beta()
        
        assert np.isclose(results['measured_beta'], 2/3, rtol=0.05)


class TestHolographicPrinciple:
    """Test Holographic calculations."""
    
    def test_beta_derivation(self):
        """Verify β = 2/3 from holography."""
        holo = HolographicPrinciple()
        results = holo.derive_beta_from_holography()
        
        assert results['beta'] == 2/3
    
    def test_rho_c_order_of_magnitude(self):
        """ρ_c^(1/4) should be ~meV scale."""
        holo = HolographicPrinciple()
        results = holo.derive_rho_c_from_uv_ir()
        
        rho_c_fourth_meV = results['rho_c_fourth_root_meV']
        # Should be within factor of 2 of 2.28 meV
        assert 1 < rho_c_fourth_meV < 5


class TestSpectralDimension:
    """Test spectral dimension calculations."""
    
    def test_uv_limit(self):
        """d_s → 2 as σ → 0."""
        sd = SpectralDimension()
        d_uv = sd.spectral_dimension(1e-10)
        assert np.isclose(d_uv, 2.0, rtol=0.01)  # Should approach 2 in UV
    
    def test_ir_limit(self):
        """d_s → 4 as σ → ∞."""
        sd = SpectralDimension()
        d_ir = sd.spectral_dimension(1e10)
        assert d_ir > 3.9
    
    def test_beta_from_ds(self):
        """Verify β = 2/3 from spectral dimension."""
        sd = SpectralDimension()
        beta = sd.beta_from_spectral_dimension()
        assert np.isclose(beta, 2/3)


class TestConvergence:
    """Test that all approaches converge to same values."""
    
    def test_alpha_convergence(self):
        """α* should be ~0.075 from AS."""
        AS = AsymptoticSafety()
        alpha_AS = AS.alpha_from_g_star()
        
        params = QGUParameters()
        alpha_expected = params.alpha_star
        
        assert np.isclose(alpha_AS, alpha_expected, rtol=0.05)
    
    def test_beta_convergence(self):
        """β should be 2/3 from multiple approaches."""
        # From LQG
        lqg = LoopQuantumGravity()
        beta_lqg = lqg.verify_beta()['measured_beta']
        
        # From Holography
        holo = HolographicPrinciple()
        beta_holo = holo.derive_beta_from_holography()['beta']
        
        # From CDT
        sd = SpectralDimension()
        beta_cdt = sd.beta_from_spectral_dimension()
        
        # All should be ~2/3
        assert np.isclose(beta_lqg, 2/3, rtol=0.05)
        assert beta_holo == 2/3
        assert np.isclose(beta_cdt, 2/3, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
QGU-DG Comprehensive Simulation Suite
======================================

Generates all figures for the Quantum Gravity Unification paper.

Author: Hugo Hertault
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse
import os
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

# QGU-DG parameters
ALPHA_STAR = 0.075
BETA = 2/3
G_STAR = 0.816
LAMBDA_STAR = 0.193

# Output directory
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, name, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    for fmt in formats:
        filepath = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        fig.savefig(filepath, format=fmt, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")


# ============================================================
# FIGURE 1: QGU UNIFICATION DIAGRAM
# ============================================================

def plot_unification_diagram():
    """Create the central unification diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors = {
        'AS': '#E74C3C', 'LQG': '#3498DB', 'Strings': '#2ECC71',
        'CDT': '#9B59B6', 'Holo': '#F39C12', 'DG': '#1A1A2E'
    }
    
    # Central box
    center = FancyBboxPatch((-0.4, -0.25), 0.8, 0.5, boxstyle="round,pad=0.05",
                            facecolor=colors['DG'], edgecolor='white', linewidth=3)
    ax.add_patch(center)
    ax.text(0, 0.05, 'Dark Geometry', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    ax.text(0, -0.12, r'$m^2_{\rm eff} = (\alpha^* M_{\rm Pl})^2[1-(\rho/\rho_c)^{2/3}]$',
            ha='center', va='center', fontsize=11, color='white')
    
    # Five approaches
    angles = [90, 162, 234, 306, 18]
    labels = [
        ('Asymptotic\nSafety', r'$g^* = 0.816$', 'AS'),
        ('Loop Quantum\nGravity', r'$A \propto V^{2/3}$', 'LQG'),
        ('String\nTheory', r'Dilaton $= \phi_{\rm DG}$', 'Strings'),
        ('CDT', r'$d_s: 4 \to 2$', 'CDT'),
        ('Holographic\nPrinciple', r'$\rho_c = \sqrt{E_{\rm Pl} E_H}$', 'Holo')
    ]
    
    for angle, (name, formula, key) in zip(angles, labels):
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        
        circle = Circle((x, y), 0.25, facecolor=colors[key], 
                        edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y + 0.05, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        
        fx = x + 0.38 * np.cos(np.radians(angle))
        fy = y + 0.38 * np.sin(np.radians(angle))
        ax.text(fx, fy, formula, ha='center', va='center',
                fontsize=9, color=colors[key], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.arrow(x - 0.2*np.cos(np.radians(angle)), 
                 y - 0.2*np.sin(np.radians(angle)),
                 -0.3*np.cos(np.radians(angle)), 
                 -0.3*np.sin(np.radians(angle)),
                 head_width=0.05, head_length=0.03, fc=colors[key], ec=colors[key])
    
    ax.text(0, 1.4, 'Quantum Gravity Unification via Dark Geometry',
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    summary = r"$\alpha^* = 0.075$ | $\beta = 2/3$ | $\rho_c^{1/4} = 2.28$ meV"
    ax.text(0, -1.3, summary, ha='center', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    save_figure(fig, 'fig01_unification_diagram')
    plt.close()


# ============================================================
# FIGURE 2: RG FLOW
# ============================================================

def plot_as_rg_flow():
    """Plot RG flow to UV fixed point."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    g_range = np.linspace(0.01, 1.5, 25)
    lam_range = np.linspace(-0.2, 0.4, 25)
    G, LAM = np.meshgrid(g_range, lam_range)
    
    # Simplified beta functions
    U = (2 - 2*G/(1 + G/np.pi)) * G
    V = (-2 - 2*G/(1 + G/np.pi)) * LAM + G/(4*np.pi*(1-2*LAM+0.01)) * 4.5
    
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 0.1)
    V_norm = V / (magnitude + 0.1)
    
    ax.quiver(G, LAM, U_norm, V_norm, magnitude, cmap='viridis', alpha=0.7)
    ax.plot(G_STAR, LAMBDA_STAR, 'r*', markersize=20, 
            label=f'UV Fixed Point $(g^*, \\lambda^*) = ({G_STAR}, {LAMBDA_STAR})$')
    ax.plot(0, 0, 'ko', markersize=10, label='Gaussian FP')
    
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label=r'$\lambda = 1/2$')
    
    ax.annotate(f'$\\alpha^* = {ALPHA_STAR}$', xy=(G_STAR, LAMBDA_STAR), 
                xytext=(1.1, 0.3), fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel(r'$g = G k^2$', fontsize=12)
    ax.set_ylabel(r'$\lambda = \Lambda/k^2$', fontsize=12)
    ax.set_title('Asymptotic Safety: RG Flow', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.2, 0.45)
    
    save_figure(fig, 'fig02_as_rg_flow')
    plt.close()


# ============================================================
# FIGURE 3: LQG SPECTRUM
# ============================================================

def plot_lqg_spectrum():
    """Plot LQG area-volume relation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    gamma = 0.2375
    j_values = np.arange(0.5, 50.5, 0.5)
    
    A_j = 8 * np.pi * gamma * np.sqrt(j_values * (j_values + 1))
    V_j = gamma**(3/2) * j_values**(3/2)
    
    axes[0].bar(j_values[:20], A_j[:20], width=0.4, color='steelblue', alpha=0.7)
    axes[0].set_xlabel(r'Spin $j$')
    axes[0].set_ylabel(r'Area $A_j$ [$\ell_{\rm Pl}^2$]')
    axes[0].set_title('LQG Area Spectrum')
    
    axes[1].bar(j_values[:20], V_j[:20], width=0.4, color='forestgreen', alpha=0.7)
    axes[1].set_xlabel(r'Spin $j$')
    axes[1].set_ylabel(r'Volume $V_j$ [$\ell_{\rm Pl}^3$]')
    axes[1].set_title('LQG Volume Spectrum')
    
    V_23 = V_j**(2/3)
    axes[2].scatter(V_23, A_j, c=j_values, cmap='plasma', s=50, alpha=0.7)
    coeffs = np.polyfit(V_23, A_j, 1)
    V_fit = np.linspace(V_23.min(), V_23.max(), 100)
    axes[2].plot(V_fit, np.poly1d(coeffs)(V_fit), 'r--', linewidth=2,
                 label=f'$A = {coeffs[0]:.2f} \\times V^{{2/3}}$')
    axes[2].set_xlabel(r'$V^{2/3}$')
    axes[2].set_ylabel(r'Area $A$')
    axes[2].set_title(r'$A \propto V^{2/3} \Rightarrow \beta = 2/3$')
    axes[2].legend()
    
    plt.tight_layout()
    save_figure(fig, 'fig03_lqg_spectrum')
    plt.close()


# ============================================================
# FIGURE 4: SPECTRAL DIMENSION
# ============================================================

def plot_spectral_dimension():
    """Plot spectral dimension flow."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sigma = np.logspace(-2, 3, 500)
    d_s = 4 * sigma / (sigma + 1)
    
    axes[0].semilogx(sigma, d_s, 'b-', linewidth=2.5, label=r'$d_s(\sigma)$')
    axes[0].axhline(4, color='green', linestyle='--', alpha=0.7, label='IR: $d_s = 4$')
    axes[0].axhline(2, color='red', linestyle='--', alpha=0.7, label='UV: $d_s = 2$')
    axes[0].fill_between(sigma, 0, d_s, where=(sigma < 1), alpha=0.2, color='red')
    axes[0].fill_between(sigma, 0, d_s, where=(sigma > 1), alpha=0.2, color='green')
    axes[0].set_xlabel(r'Diffusion time $\sigma$')
    axes[0].set_ylabel(r'Spectral dimension $d_s$')
    axes[0].set_title('CDT: Spectral Dimension Flow')
    axes[0].legend()
    axes[0].set_ylim(0, 5)
    
    zeta = np.logspace(-2, 2, 500)
    d_eff = 2 + 2 * zeta**2 / (zeta**2 + 1)
    
    axes[1].semilogx(zeta, d_eff, 'purple', linewidth=2.5)
    axes[1].axhline(4, color='green', linestyle='--', alpha=0.7, label='IR')
    axes[1].axhline(2, color='red', linestyle='--', alpha=0.7, label='UV')
    axes[1].set_xlabel(r'Holographic coordinate $\zeta$')
    axes[1].set_ylabel(r'Effective dimension $d_{\rm eff}$')
    axes[1].set_title('DG: Effective Dimension')
    axes[1].legend()
    axes[1].set_ylim(0, 5)
    
    plt.tight_layout()
    save_figure(fig, 'fig04_spectral_dimension')
    plt.close()


# ============================================================
# FIGURE 5: UV-IR CONNECTION
# ============================================================

def plot_uv_ir_connection():
    """Visualize UV-IR connection."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    E_Pl = 28  # log10(eV)
    E_H = -33  # log10(eV)
    E_c = (E_Pl + E_H) / 2
    
    ax.axhline(E_Pl, color='red', linewidth=3, label=r'$E_{\rm Pl} = 10^{28}$ eV (UV)')
    ax.axhline(E_H, color='blue', linewidth=3, label=r'$E_H = 10^{-33}$ eV (IR)')
    ax.axhline(E_c, color='green', linewidth=4, label=r'$\sqrt{E_{\rm Pl} E_H} \sim 10^{-2.5}$ eV')
    
    ax.fill_between([0, 10], E_Pl-2, E_Pl+2, alpha=0.1, color='red')
    ax.fill_between([0, 10], E_H-2, E_H+2, alpha=0.1, color='blue')
    ax.fill_between([0, 10], E_c-1, E_c+1, alpha=0.3, color='green')
    
    ax.annotate(r'$\rho_c^{1/4} = \sqrt{E_{\rm Pl} E_H}/2 \approx 2.28$ meV',
                xy=(5, E_c), xytext=(2, -5), fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-36, 30)
    ax.set_ylabel(r'$\log_{10}(E/{\rm eV})$', fontsize=12)
    ax.set_title('UV-IR Connection: Origin of $\\rho_c$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xticks([])
    
    save_figure(fig, 'fig05_uv_ir_connection')
    plt.close()


# ============================================================
# FIGURE 6: EFFECTIVE MASS
# ============================================================

def plot_effective_mass():
    """Plot effective mass function."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rho = np.logspace(-3, 3, 1000)
    m2_eff = 1 - rho**(2/3)
    
    axes[0].semilogx(rho, m2_eff, 'b-', linewidth=2.5)
    axes[0].axhline(0, color='k', linewidth=1)
    axes[0].axvline(1, color='gray', linestyle='--', alpha=0.7, label=r'$\rho_c$')
    axes[0].fill_between(rho, m2_eff, 0, where=(m2_eff > 0), alpha=0.2, color='green', label='DE')
    axes[0].fill_between(rho, m2_eff, 0, where=(m2_eff < 0), alpha=0.2, color='red', label='DM')
    axes[0].set_xlabel(r'$\rho/\rho_c$')
    axes[0].set_ylabel(r'$m^2_{\rm eff} / (\alpha^* M_{\rm Pl})^2$')
    axes[0].set_title('Effective Mass Squared')
    axes[0].legend()
    axes[0].set_ylim(-5, 1.5)
    
    x = rho
    w = -1 / (1 + np.exp(10 * (x - 1)))
    
    axes[1].semilogx(rho, w, 'purple', linewidth=2.5)
    axes[1].axhline(-1, color='green', linestyle='--', alpha=0.7, label='$w = -1$')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.7, label='$w = 0$')
    axes[1].axhline(-1/3, color='orange', linestyle=':', alpha=0.7, label='Acceleration')
    axes[1].set_xlabel(r'$\rho/\rho_c$')
    axes[1].set_ylabel(r'$w$')
    axes[1].set_title('Equation of State')
    axes[1].legend()
    axes[1].set_ylim(-1.2, 0.2)
    
    plt.tight_layout()
    save_figure(fig, 'fig06_effective_mass')
    plt.close()


# ============================================================
# FIGURE 7: HALO PROFILES
# ============================================================

def plot_halo_profiles():
    """Compare NFW and DG halo profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    r = np.logspace(-1, 2.5, 200)
    r_s = 20
    x = r / r_s
    
    rho_nfw = 1 / (x * (1 + x)**2)
    
    r_core = ALPHA_STAR * r_s
    x_core = r / r_core
    core_factor = x_core**2 / (1 + x_core**2)
    rho_dg = rho_nfw * (0.1 + 0.9 * core_factor)
    
    axes[0].loglog(r, rho_nfw, 'r-', linewidth=2.5, label='NFW (ΛCDM)')
    axes[0].loglog(r, rho_dg, 'b-', linewidth=2.5, label='DG (cored)')
    axes[0].axvline(r_s, color='gray', linestyle=':', alpha=0.5, label=f'$r_s$')
    axes[0].axvline(r_core, color='blue', linestyle=':', alpha=0.5, label=f'$r_{{core}}$')
    axes[0].set_xlabel('Radius $r$ [kpc]')
    axes[0].set_ylabel(r'$\rho/\rho_s$')
    axes[0].set_title('Halo Density Profiles')
    axes[0].legend()
    axes[0].set_xlim(0.1, 300)
    axes[0].set_ylim(1e-6, 10)
    
    d_log_rho_nfw = np.gradient(np.log10(rho_nfw + 1e-10), np.log10(r))
    d_log_rho_dg = np.gradient(np.log10(rho_dg + 1e-10), np.log10(r))
    
    axes[1].semilogx(r, d_log_rho_nfw, 'r-', linewidth=2.5, label='NFW')
    axes[1].semilogx(r, d_log_rho_dg, 'b-', linewidth=2.5, label='DG')
    axes[1].axhline(-1, color='red', linestyle='--', alpha=0.7, label='Cusp ($n=-1$)')
    axes[1].axhline(0, color='blue', linestyle='--', alpha=0.7, label='Core ($n=0$)')
    axes[1].set_xlabel('Radius $r$ [kpc]')
    axes[1].set_ylabel(r'Slope $d\log\rho/d\log r$')
    axes[1].set_title('Inner Density Slope')
    axes[1].legend()
    axes[1].set_xlim(0.1, 100)
    axes[1].set_ylim(-3.5, 0.5)
    
    plt.tight_layout()
    save_figure(fig, 'fig07_halo_profiles')
    plt.close()


# ============================================================
# FIGURE 8: POWER SPECTRUM
# ============================================================

def plot_power_spectrum():
    """Plot power spectrum with DG suppression."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k = np.logspace(-3, 1, 500)
    k_eq = 0.01
    P_LCDM = k * (1 + (k/k_eq)**2)**(-2)
    P_LCDM = P_LCDM / P_LCDM.max() * 1e4
    
    k_s = 0.3
    S_k = 1 - 0.25 * (1 - 1 / (1 + (k / k_s)**2.8))
    P_DG = P_LCDM * S_k
    
    axes[0].loglog(k, P_LCDM, 'r-', linewidth=2.5, label='ΛCDM')
    axes[0].loglog(k, P_DG, 'b-', linewidth=2.5, label='Dark Geometry')
    axes[0].axvline(k_s, color='gray', linestyle='--', alpha=0.7, label=f'$k_s = {k_s}$')
    axes[0].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0].set_ylabel(r'$P(k)$')
    axes[0].set_title('Matter Power Spectrum')
    axes[0].legend()
    axes[0].set_xlim(1e-3, 10)
    
    axes[1].semilogx(k, S_k, 'b-', linewidth=2.5, label='DG/ΛCDM ratio')
    axes[1].axhline(1, color='r', linestyle='--', alpha=0.7, label='ΛCDM')
    axes[1].axhline(0.75, color='gray', linestyle=':', alpha=0.7, label='25% suppression')
    axes[1].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[1].set_ylabel(r'$P_{\rm DG}/P_{\Lambda{\rm CDM}}$')
    axes[1].set_title('DG Suppression Function')
    axes[1].legend()
    axes[1].set_xlim(1e-3, 10)
    axes[1].set_ylim(0.7, 1.05)
    
    plt.tight_layout()
    save_figure(fig, 'fig08_power_spectrum')
    plt.close()


# ============================================================
# FIGURE 9: σ₈ TENSION
# ============================================================

def plot_s8_tension():
    """Show σ₈ tension resolution."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    measurements = [
        ('Planck CMB', 0.811, 0.006, 'red', 's'),
        ('DES Y3', 0.759, 0.023, 'blue', 'o'),
        ('KiDS-1000', 0.766, 0.020, 'green', '^'),
        ('HSC Y1', 0.780, 0.030, 'purple', 'D'),
    ]
    
    for i, (name, val, err, color, marker) in enumerate(measurements):
        ax.errorbar(val, i, xerr=err, fmt=marker, color=color,
                    markersize=12, capsize=5, capthick=2, linewidth=2, label=name)
    
    ax.axvline(0.811, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ΛCDM')
    ax.axvspan(0.811 - 0.006, 0.811 + 0.006, alpha=0.1, color='red')
    
    ax.axvline(0.76, color='blue', linestyle='-', linewidth=3, alpha=0.7, label='DG')
    ax.axvspan(0.74, 0.78, alpha=0.2, color='blue')
    
    ax.set_xlabel(r'$\sigma_8$', fontsize=14)
    ax.set_yticks(range(len(measurements)))
    ax.set_yticklabels([m[0] for m in measurements])
    ax.set_title(r'$\sigma_8$ Tension Resolution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0.65, 0.90)
    
    save_figure(fig, 'fig09_s8_tension')
    plt.close()


# ============================================================
# FIGURE 10: w(z) EVOLUTION
# ============================================================

def plot_w_evolution():
    """Plot w(z) evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    z = np.linspace(0, 2, 200)
    
    w_LCDM = -1 * np.ones_like(z)
    w_0, w_a = -0.95, 0.25
    w_DG = w_0 + w_a * z / (1 + z)
    
    axes[0].plot(z, w_LCDM, 'r--', linewidth=2.5, label='ΛCDM')
    axes[0].plot(z, w_DG, 'b-', linewidth=2.5, label='DG')
    axes[0].axhline(-1/3, color='orange', linestyle=':', alpha=0.7, label='Acceleration')
    axes[0].set_xlabel('Redshift $z$')
    axes[0].set_ylabel('$w(z)$')
    axes[0].set_title('Dark Energy Equation of State')
    axes[0].legend()
    axes[0].set_ylim(-1.3, 0)
    
    ax2 = axes[1]
    ax2.plot(-1, 0, 'r*', markersize=20, label='ΛCDM')
    ax2.plot(w_0, w_a, 'b*', markersize=20, label='DG')
    
    ell = Ellipse((-0.7, -1.0), 0.6, 1.5, angle=30,
                  facecolor='green', alpha=0.3, edgecolor='green', linewidth=2)
    ax2.add_patch(ell)
    ax2.plot(-0.7, -1.0, 'g^', markersize=12, label='DESI hint')
    
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(-1, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('$w_0$')
    ax2.set_ylabel('$w_a$')
    ax2.set_title('$w_0$-$w_a$ Parameter Space')
    ax2.legend()
    ax2.set_xlim(-1.5, -0.3)
    ax2.set_ylim(-2.5, 1)
    
    plt.tight_layout()
    save_figure(fig, 'fig10_w_evolution')
    plt.close()


# ============================================================
# FIGURE 11: PREDICTIONS
# ============================================================

def plot_predictions():
    """Summary of predictions."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'QGU-DG: Testable Predictions',
            transform=ax.transAxes, ha='center', fontsize=18, fontweight='bold')
    
    predictions = [
        ('σ₈ Tension', '0.74-0.78', '0.811', '0.76 (WL)', '✓'),
        ('Dwarf Cores', 'n ≈ 0', 'n = -1', 'n ≈ 0', '✓'),
        ('MW Satellites', '~60', '~500', '~60', '✓'),
        ('w(z)', 'Evolving', 'w = -1', 'Hints', '~'),
        ('P(k) Suppression', 'Yes', 'No', 'Testing', '→'),
    ]
    
    y_start = 0.8
    headers = ['Observable', 'DG', 'ΛCDM', 'Observed', '']
    x_pos = [0.05, 0.3, 0.5, 0.7, 0.9]
    
    for x, h in zip(x_pos, headers):
        ax.text(x, y_start + 0.05, h, transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    for i, row in enumerate(predictions):
        y = y_start - (i + 1) * 0.1
        colors = ['black', 'blue', 'red', 'black', 'green']
        for x, val, col in zip(x_pos, row, colors):
            ax.text(x, y, val, transform=ax.transAxes, fontsize=11, color=col)
    
    ax.text(0.5, 0.15, 'Repository: github.com/HugoHertault/QGU-DG',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')
    
    save_figure(fig, 'fig11_predictions')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def generate_all_figures():
    """Generate all figures."""
    print("=" * 60)
    print("QGU-DG Figure Generation Suite")
    print("=" * 60)
    
    figures = [
        ("Fig 1: Unification Diagram", plot_unification_diagram),
        ("Fig 2: AS RG Flow", plot_as_rg_flow),
        ("Fig 3: LQG Spectrum", plot_lqg_spectrum),
        ("Fig 4: Spectral Dimension", plot_spectral_dimension),
        ("Fig 5: UV-IR Connection", plot_uv_ir_connection),
        ("Fig 6: Effective Mass", plot_effective_mass),
        ("Fig 7: Halo Profiles", plot_halo_profiles),
        ("Fig 8: Power Spectrum", plot_power_spectrum),
        ("Fig 9: σ₈ Tension", plot_s8_tension),
        ("Fig 10: w(z) Evolution", plot_w_evolution),
        ("Fig 11: Predictions", plot_predictions),
    ]
    
    for name, func in figures:
        print(f"\n{name}")
        try:
            func()
            print("  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()

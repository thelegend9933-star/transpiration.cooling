import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
import os

# Create output directory
output_dir = "output_2d"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# GEOMETRY & FLUID PROPERTIES (Air at 293K)
# =============================================================================
W = 0.14          # Porous plate width [m]
L_thickness = 0.005 # Porous plate thickness [m]
D_h = 0.14        # Channel hydraulic diameter [m]
L_chan = 0.60     # Channel length [m]

rho = 1.204       # Fluid density [kg/m^3]
mu = 1.825e-5     # Fluid dynamic viscosity [Pa.s]
k_f = 0.02551     # Fluid thermal conductivity [W/m.K]
Pr = 0.713        # Prandtl number
cp = 1007.0       # Fluid specific heat [J/kg.K]
T_0 = 293.0       # Inlet coolant temperature [K]
Q_pp = 38300.0    # Imposed heat flux [W/m^2]

# Solver Grid
Nz = 50
Nx = 30
dz = L_thickness / (Nz - 1)
dx = W / (Nx - 1)
x_coords = np.linspace(0, W, Nx)
z_coords = np.linspace(0, L_thickness, Nz)
X, Z = np.meshgrid(x_coords, z_coords)

# =============================================================================
# SPECIMEN DEFINITIONS & EXPERIMENTAL DATA
# =============================================================================
specimens = {
    'SS316L_mesh_60':  {'type': 'mesh',   'dp': 60e-6,  'eps': 0.379, 'K_z': 0.441e-12, 'ks': 16.3, 'AR': 3.5, 'Re_exp': [39000, 63000, 70000], 'eta_exp': [0.54, 0.60, 0.64]},
    'Ti_mesh_60':      {'type': 'mesh',   'dp': 60e-6,  'eps': 0.379, 'K_z': 0.441e-12, 'ks': 21.9, 'AR': 3.5, 'Re_exp': [39000, 63000, 70000], 'eta_exp': [0.77, 0.83, 0.88]},
    'SS316L_powder_60':{'type': 'powder', 'dp': 60e-6,  'eps': 0.350, 'K_z': 2.03e-12,  'ks': 16.3, 'AR': 1.0, 'Re_exp': [38000, 62000, 67000], 'eta_exp': [0.58, 0.69, 0.76]},
    'SS316L_mesh_80':  {'type': 'mesh',   'dp': 80e-6,  'eps': 0.379, 'K_z': 0.782e-12, 'ks': 16.3, 'AR': 3.5, 'Re_exp': [38000, 62000, 67000], 'eta_exp': [0.567, 0.633, 0.677]},
    'SS316L_mesh_100': {'type': 'mesh',   'dp': 100e-6, 'eps': 0.379, 'K_z': 1.221e-12, 'ks': 16.3, 'AR': 3.5, 'Re_exp': [38000, 62000, 67000], 'eta_exp': [0.556, 0.621, 0.657]},
}

# =============================================================================
# CORRELATIONS & GOVERNING EQUATIONS
# =============================================================================
def calc_u_from_Re(Re):
    """Calculate mean coolant bulk velocity based on channel Re."""
    return Re * mu / (rho * D_h)

def get_h_sf_and_a_sf(spec_name, u_mean):
    s = specimens[spec_name]
    Re_p = rho * (u_mean / s['eps']) * s['dp'] / mu
    Nu_sf = 2.0 + 1.1 * (Re_p**0.6) * (Pr**(1/3))
    h_sf = Nu_sf * k_f / s['dp']
    
    if s['type'] == 'mesh':
        a_sf = 4 * s['eps'] / s['dp']
    else:
        a_sf = 6 * (1 - s['eps']) / s['dp']
    return h_sf, a_sf

def calc_eta(T_s_mean_hot, Re, u_avg, s, C_F):
    h_conv = Q_pp / (T_s_mean_hot - T_0)
    Nu = h_conv * D_h / k_f
    Nu_0 = 0.023 * (Re**0.8) * (Pr**0.4)
    f_0 = 0.316 * (Re**(-0.25))
    
    dP = (mu / s['K_z'] * u_avg + rho * C_F / np.sqrt(s['K_z']) * (u_avg**2)) * L_thickness
    f_model = 2 * dP * D_h / (rho * (u_avg**2) * L_chan)
    
    return (Nu / Nu_0) / ((f_model / f_0)**(1/3))

# =============================================================================
# 2D FLOW FIELD SOLVER (Darcy Pressure Solver)
# =============================================================================
def solve_flow_field(spec_name, Re):
    """
    Solves for the 2D pressure field P(x,z) using Darcy's law to allow for lateral flow.
    Assumes a slightly parabolic non-uniform inlet flow from the channel, which
    drives lateral pressure gradients. Anisotropy (K_x > K_z) allows mesh to spread
    this flow laterally much better than isotropic powder.
    """
    s = specimens[spec_name]
    K_z = s['K_z']
    K_x = K_z * s['AR']
    u_avg = calc_u_from_Re(Re)
    
    # Introduce a 20% deficit profile at edges to simulate channel flow
    u_in = u_avg * (1.0 - 0.2 * (2*x_coords/W - 1)**2)
    u_in = u_in * (u_avg / np.mean(u_in)) # Maintain strict mass conservation
    
    N = Nz * Nx
    A_p = sp.lil_matrix((N, N))
    b_p = np.zeros(N)
    
    def idx(i, j): return i * Nx + j
    
    for i in range(Nz):
        for j in range(Nx):
            row = idx(i, j)
            if i == Nz - 1:
                # Outlet P = 0
                A_p[row, row] = 1.0
                b_p[row] = 0.0
            elif i == 0:
                # Inlet fixed flow: P_0 - P_1 = (mu/K_z)*u_in*dz
                A_p[row, row] = 1.0
                A_p[row, idx(1, j)] = -1.0
                b_p[row] = (mu / K_z) * u_in[j] * dz
            elif j == 0 and 0 < i < Nz - 1:
                # Left Wall Adibatic (dP/dx = 0)
                A_p[row, row] = 1.0
                A_p[row, idx(i, 1)] = -1.0
                b_p[row] = 0.0
            elif j == Nx - 1 and 0 < i < Nz - 1:
                # Right Wall Adibatic (dP/dx = 0)
                A_p[row, row] = 1.0
                A_p[row, idx(i, Nx - 2)] = -1.0
                b_p[row] = 0.0
            else:
                # Internal Nodes: K_x * d2P/dx2 + K_z * d2P/dz2 = 0
                A_p[row, row] = -2*K_x/(dx**2) - 2*K_z/(dz**2)
                A_p[row, idx(i, j-1)] = K_x/(dx**2)
                A_p[row, idx(i, j+1)] = K_x/(dx**2)
                A_p[row, idx(i-1, j)] = K_z/(dz**2)
                A_p[row, idx(i+1, j)] = K_z/(dz**2)
                b_p[row] = 0.0
                
    # Solve Pressure
    P = spsolve(A_p.tocsr(), b_p).reshape((Nz, Nx))
    u_x = np.zeros((Nz, Nx))
    u_z = np.zeros((Nz, Nx))
    
    # Compute u_x and u_z via central differences
    for i in range(Nz):
        for j in range(Nx):
            if i == 0:
                u_z[i, j] = u_in[j]
            elif i == Nz - 1:
                u_z[i, j] = -(K_z / mu) * (P[i, j] - P[i-1, j]) / dz
            else:
                u_z[i, j] = -(K_z / mu) * (P[i+1, j] - P[i-1, j]) / (2*dz)
                
            if j == 0 or j == Nx - 1:
                u_x[i, j] = 0.0
            else:
                u_x[i, j] = -(K_x / mu) * (P[i, j+1] - P[i, j-1]) / (2*dx)
                
    return u_z, u_x, u_avg

# =============================================================================
# 2D LTNE FINITE DIFFERENCE SOLVER
# =============================================================================
def solve_2d_ltne(spec_name, Re, C_F):
    s = specimens[spec_name]
    eps = s['eps']
    ks = s['ks']
    
    # 1. Acquire true physical velocity vectors based on Anisotropy
    u_z, u_x, u_avg = solve_flow_field(spec_name, Re)
    u_mean_mag = np.mean(np.sqrt(u_z**2 + u_x**2))
    
    h_sf, a_sf = get_h_sf_and_a_sf(spec_name, u_mean_mag)
    
    N = Nz * Nx
    A = sp.lil_matrix((2*N, 2*N))
    b = np.zeros(2*N)
    
    # Diffusion coefficients (Solid is strictly isotropic)
    diff_f_z = eps * k_f / (dz**2)
    diff_f_x = eps * k_f / (dx**2)
    diff_s_z = (1 - eps) * ks / (dz**2)
    diff_s_x = (1 - eps) * ks / (dx**2)
    coup = h_sf * a_sf

    def idx(i, j, is_solid=False):
        offset = N if is_solid else 0
        return offset + i * Nx + j

    for i in range(Nz):
        for j in range(Nx):
            row_f = idx(i, j, False)
            row_s = idx(i, j, True)
            
            # ---------------- BOUNDARY CONDITIONS: FLUID ----------------
            if i == 0:
                # z=0 Inlet: Applies to all j (including corners)
                A[row_f, row_f] = 1.0
                b[row_f] = T_0
            elif i == Nz - 1:
                # z=L Outlet: Applies to all j (including corners)
                A[row_f, row_f] = 1.0
                A[row_f, idx(i-1, j, False)] = -1.0
                b[row_f] = 0.0
            elif j == 0 and 0 < i < Nz - 1:
                # x=0: dTf/dx = 0 (Internal Z nodes only to avoid corner conflict)
                A[row_f, row_f] = 1.0
                A[row_f, idx(i, j+1, False)] = -1.0
                b[row_f] = 0.0
            elif j == Nx - 1 and 0 < i < Nz - 1:
                # x=W: dTf/dx = 0
                A[row_f, row_f] = 1.0
                A[row_f, idx(i, j-1, False)] = -1.0
                b[row_f] = 0.0
            else:
                # Internal Fluid with 2D Upwind Convection
                A[row_f, row_f] = 2*diff_f_z + 2*diff_f_x + coup
                A[row_f, idx(i-1, j, False)] = -diff_f_z
                A[row_f, idx(i+1, j, False)] = -diff_f_z
                A[row_f, idx(i, j-1, False)] = -diff_f_x
                A[row_f, idx(i, j+1, False)] = -diff_f_x
                A[row_f, row_s] = -coup
                
                # Z-Convection (Upwind)
                uz_val = u_z[i, j]
                if uz_val >= 0:
                    c_z = eps * rho * cp * uz_val / dz
                    A[row_f, row_f] += c_z
                    A[row_f, idx(i-1, j, False)] -= c_z
                else:
                    c_z = -eps * rho * cp * uz_val / dz
                    A[row_f, row_f] += c_z
                    A[row_f, idx(i+1, j, False)] -= c_z
                    
                # X-Convection (Upwind)
                ux_val = u_x[i, j]
                if ux_val >= 0:
                    c_x = eps * rho * cp * ux_val / dx
                    A[row_f, row_f] += c_x
                    A[row_f, idx(i, j-1, False)] -= c_x
                else:
                    c_x = -eps * rho * cp * ux_val / dx
                    A[row_f, row_f] += c_x
                    A[row_f, idx(i, j+1, False)] -= c_x
                
            # ---------------- BOUNDARY CONDITIONS: SOLID ----------------
            if i == 0:
                # z=0: dTs/dz = 0
                A[row_s, row_s] = 1.0
                A[row_s, idx(i+1, j, True)] = -1.0
                b[row_s] = 0.0
            elif i == Nz - 1:
                # z=L: Constant heat flux
                A[row_s, row_s] = 1.0
                A[row_s, idx(i-1, j, True)] = -1.0
                b[row_s] = (Q_pp / ((1 - eps) * ks)) * dz
            elif j == 0 and 0 < i < Nz - 1:
                # x=0: dTs/dx = 0
                A[row_s, row_s] = 1.0
                A[row_s, idx(i, j+1, True)] = -1.0
                b[row_s] = 0.0
            elif j == Nx - 1 and 0 < i < Nz - 1:
                # x=W: dTs/dx = 0
                A[row_s, row_s] = 1.0
                A[row_s, idx(i, j-1, True)] = -1.0
                b[row_s] = 0.0
            else:
                # Internal Solid
                A[row_s, row_s] = 2*diff_s_z + 2*diff_s_x + coup
                A[row_s, idx(i-1, j, True)] = -diff_s_z
                A[row_s, idx(i+1, j, True)] = -diff_s_z
                A[row_s, idx(i, j-1, True)] = -diff_s_x
                A[row_s, idx(i, j+1, True)] = -diff_s_x
                A[row_s, row_f] = -coup

    A = A.tocsr()
    x = spsolve(A, b)
    
    T_f = x[:N].reshape((Nz, Nx))
    T_s = x[N:].reshape((Nz, Nx))
    
    return T_f, T_s, u_avg

# =============================================================================
# CALIBRATION & VALIDATION
# =============================================================================
C_F_powder = 0.143

def calibration_objective(C_F_guess):
    s_name = 'SS316L_mesh_60'
    error = 0
    for Re, eta_exp in zip(specimens[s_name]['Re_exp'], specimens[s_name]['eta_exp']):
        _, T_s, u_avg = solve_2d_ltne(s_name, Re, C_F_guess)
        eta_mod = calc_eta(np.mean(T_s[-1, :]), Re, u_avg, specimens[s_name], C_F_guess)
        error += (eta_mod - eta_exp)**2
    return error

print("Starting Calibration on SS316L_mesh_60...")
res = minimize_scalar(calibration_objective, bounds=(0.001, 2.0), method='bounded')
C_F_calibrated = res.x
print(f"Calibrated C_F_mesh = {C_F_calibrated:.4f}\n")

print("-" * 80)
print(f"{'Specimen':<18} {'Type':<8} {'dp (um)':<8} {'eps':<6} {'K_z (m2)':<12} {'K_x (m2)':<12} {'AR':<4}")
print("-" * 80)
for name, s in specimens.items():
    print(f"{name:<18} {s['type']:<8} {s['dp']*1e6:<8.0f} {s['eps']:<6.3f} {s['K_z']:<12.3e} {s['K_z']*s['AR']:<12.3e} {s['AR']:<4.1f}")
print("-" * 80, "\n")

results = []
print(f"{'Specimen':<18} {'Re':<8} {'eta_exp':<8} {'eta_mod':<8} {'Error %':<8}")
print("-" * 80)
for name, s in specimens.items():
    C_F = C_F_powder if s['type'] == 'powder' else C_F_calibrated
    for Re, eta_exp in zip(s['Re_exp'], s['eta_exp']):
        _, T_s, u_avg = solve_2d_ltne(name, Re, C_F)
        eta_mod = calc_eta(np.mean(T_s[-1, :]), Re, u_avg, s, C_F)
        err_pct = (eta_mod - eta_exp) / eta_exp * 100
        results.append((name, Re, eta_exp, eta_mod))
        print(f"{name:<18} {Re:<8} {eta_exp:<8.3f} {eta_mod:<8.3f} {err_pct:>6.1f}%")

print("\nSpatial Uniformity (sigma of Ts at hot face):")
for name, s in specimens.items():
    Re_peak = s['Re_exp'][-1]
    C_F = C_F_powder if s['type'] == 'powder' else C_F_calibrated
    _, T_s, _ = solve_2d_ltne(name, Re_peak, C_F)
    sigma_predicted = np.std(T_s[-1, :])
    print(f"  {name:<16}: sigma_Ts = {sigma_predicted:.3f} K")

# =============================================================================
# GENERATE FIGURES
# =============================================================================
# Figure 1: 2D Contour Maps of Ts at peak Re
fig1, axes1 = plt.subplots(1, 5, figsize=(20, 5))
for idx, name in enumerate(specimens.keys()):
    Re_peak = specimens[name]['Re_exp'][-1]
    C_F = C_F_powder if specimens[name]['type'] == 'powder' else C_F_calibrated
    _, T_s, _ = solve_2d_ltne(name, Re_peak, C_F)
    
    c = axes1[idx].contourf(X*100, Z*1000, T_s, levels=20, cmap='inferno')
    axes1[idx].set_title(f"{name}\nRe={Re_peak}")
    axes1[idx].set_xlabel("x (cm)")
    if idx == 0: axes1[idx].set_ylabel("z (mm)")
    fig1.colorbar(c, ax=axes1[idx], orientation='horizontal', pad=0.15)
fig1.suptitle("Figure 1: 2D Solid Temperature Contours ($T_s$) at Peak Re", fontsize=16)
fig1.savefig(f"{output_dir}/Figure1_Contours.png", dpi=200, bbox_inches='tight')

# Figure 2: Lateral profile Ts(x) at z=L
fig2, ax2 = plt.subplots(figsize=(8, 6))
for name in specimens.keys():
    Re_peak = specimens[name]['Re_exp'][-1]
    C_F = C_F_powder if specimens[name]['type'] == 'powder' else C_F_calibrated
    _, T_s, _ = solve_2d_ltne(name, Re_peak, C_F)
    ax2.plot(x_coords, T_s[-1, :], label=f"{name}")

for x_meas in [0.035, 0.07, 0.105]:
    ax2.axvline(x_meas, color='k', linestyle='--', alpha=0.5)

ax2.set_title("Figure 2: Lateral Profile $T_s(x)$ at Hot Face ($z=L$) at Peak Re")
ax2.set_xlabel("Width $x$ (m)")
ax2.set_ylabel("Solid Temperature $T_s$ (K)")
ax2.legend()
ax2.grid(True)
fig2.savefig(f"{output_dir}/Figure2_LateralProfile.png", dpi=200)

# Figure 3: Parity Plot
fig3, ax3 = plt.subplots(figsize=(6, 6))
eta_exps = [r[2] for r in results]
eta_mods = [r[3] for r in results]

rmse = np.sqrt(np.mean((np.array(eta_mods) - np.array(eta_exps))**2))
mae = np.mean(np.abs(np.array(eta_mods) - np.array(eta_exps)))

ax3.scatter(eta_exps, eta_mods, color='blue', edgecolors='k')
lims = [0.4, 1.0]
ax3.plot(lims, lims, 'k-', alpha=0.75)
ax3.plot(lims, [l*1.08 for l in lims], 'r--', alpha=0.5, label='+8% Band')
ax3.plot(lims, [l*0.92 for l in lims], 'r--', alpha=0.5, label='-8% Band')
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_title(f"Figure 3: Parity Plot (RMSE: {rmse:.3f}, MAE: {mae:.3f})")
ax3.set_xlabel("Experimental $\eta$")
ax3.set_ylabel("Model Predicted $\eta$")
ax3.legend()
ax3.grid(True)
fig3.savefig(f"{output_dir}/Figure3_Parity.png", dpi=200)

# Figure 4: eta vs Re extended
Re_ext = np.linspace(35000, 100000, 15)
fig4, ax4 = plt.subplots(figsize=(8, 6))
for name in specimens.keys():
    s = specimens[name]
    C_F = C_F_powder if s['type'] == 'powder' else C_F_calibrated
    eta_ext = []
    for R in Re_ext:
        _, T_s, u_avg = solve_2d_ltne(name, R, C_F)
        eta_ext.append(calc_eta(np.mean(T_s[-1, :]), R, u_avg, s, C_F))
    
    p = ax4.plot(Re_ext, eta_ext, '-')
    ax4.scatter(s['Re_exp'], s['eta_exp'], color=p[0].get_color(), label=name, zorder=5)

ax4.set_title("Figure 4: $\eta$ vs Re (Extended to 100,000)")
ax4.set_xlabel("Reynolds Number (Re)")
ax4.set_ylabel("Cooling Effectiveness $\eta$")
ax4.legend()
ax4.grid(True)
fig4.savefig(f"{output_dir}/Figure4_EtaVsRe.png", dpi=200)

# Figure 5: eta vs AR for SS316L_mesh_60
fig5, ax5 = plt.subplots(figsize=(8, 6))
AR_range = np.linspace(1.0, 5.0, 10)
s_base = specimens['SS316L_mesh_60'].copy()

for Re in [39000, 63000, 70000]:
    eta_ar = []
    for AR in AR_range:
        specimens['SS316L_mesh_60']['AR'] = AR
        _, T_s, u_avg = solve_2d_ltne('SS316L_mesh_60', Re, C_F_calibrated)
        eta_ar.append(calc_eta(np.mean(T_s[-1, :]), Re, u_avg, specimens['SS316L_mesh_60'], C_F_calibrated))
    specimens['SS316L_mesh_60'] = s_base.copy()
    ax5.plot(AR_range, eta_ar, marker='o', label=f"Re={Re}")

ax5.set_title("Figure 5: Effect of Anisotropy Ratio (AR) on $\eta$")
ax5.set_xlabel("Anisotropy Ratio $AR = K_x / K_z$")
ax5.set_ylabel("Cooling Effectiveness $\eta$")
ax5.legend()
ax5.grid(True)
fig5.savefig(f"{output_dir}/Figure5_EtaVsAR.png", dpi=200)

# Figure 6: TRUE physical flow vectors calculated from Darcy Pressure Solver
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 5))

# Calculate accurate flow vectors for Anisotropic Mesh (AR=3.5)
u_z_mesh, u_x_mesh, _ = solve_flow_field('SS316L_mesh_60', 67000)
ax6a.streamplot(X, Z, u_x_mesh, u_z_mesh, density=1.0, color='b')
ax6a.set_title("Wire Mesh (AR=3.5)\nLateral spread mitigates edge deficits")
ax6a.set_xlabel("x (m)")
ax6a.set_ylabel("z (m)")

# Calculate accurate flow vectors for Isotropic Powder (AR=1.0)
u_z_pow, u_x_pow, _ = solve_flow_field('SS316L_powder_60', 67000)
ax6b.streamplot(X, Z, u_x_pow, u_z_pow, density=1.0, color='g')
ax6b.set_title("Powder (AR=1.0)\nFlow remains constrained to entry profile")
ax6b.set_xlabel("x (m)")

fig6.suptitle("Figure 6: True Darcy Coolant Streamlines at Re=67000", fontsize=14)
fig6.savefig(f"{output_dir}/Figure6_Streamlines.png", dpi=200)

print(f"\nAll 6 figures successfully saved to /{output_dir}")
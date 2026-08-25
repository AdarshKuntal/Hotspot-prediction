import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

df = pd.read_csv('/mnt/user-data/uploads/1775918678037_thermal_dataset_v2.csv')
FEATURES = ['power_density','thermal_cond_si','boundary_temp','metal_cond_Cu',
            'metal_cond_W','die_thickness','block_power_map','tim_hs_conductance',
            'x','y','hotspot_cx','hotspot_cy','sigma_spot','r_sq']
TARGET = 'temperature'
X = df[FEATURES].values
y = df[TARGET].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train models
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5,
                           random_state=42, n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)

mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                   solver='adam', learning_rate_init=0.001, max_iter=500,
                   early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=10, random_state=42)
mlp.fit(X_train_sc, y_train)

# ── FIXED 3D Thermal Surface ──────────────────────────────────────────────
res = 60
xg = np.linspace(0, 1, res)
yg = np.linspace(0, 1, res)
XX, YY = np.meshgrid(xg, yg)

med = np.median(X_train, axis=0).copy()

# Fixed physically meaningful hotspot parameters
hcx_fixed   = 0.5
hcy_fixed   = 0.5
sigma_fixed = 0.08   # sharp hotspot for clear peak
bpm_fixed   = 12.0   # high block power to amplify hotspot

grid_pts = np.tile(med, (res*res, 1))
idx = {f: i for i, f in enumerate(FEATURES)}
grid_pts[:, idx['x']]          = XX.ravel()
grid_pts[:, idx['y']]          = YY.ravel()
grid_pts[:, idx['hotspot_cx']] = hcx_fixed
grid_pts[:, idx['hotspot_cy']] = hcy_fixed
grid_pts[:, idx['sigma_spot']] = sigma_fixed
grid_pts[:, idx['block_power_map']] = bpm_fixed
grid_pts[:, idx['r_sq']]       = (XX.ravel() - hcx_fixed)**2 + (YY.ravel() - hcy_fixed)**2

T_rf  = rf.predict(grid_pts).reshape(res, res)

grid_pts_sc = scaler.transform(grid_pts)
T_mlp = mlp.predict(grid_pts_sc).reshape(res, res)

print(f"RF  T range: {T_rf.min():.2f} -- {T_rf.max():.2f} K")
print(f"MLP T range: {T_mlp.min():.2f} -- {T_mlp.max():.2f} K")

# ── Figure: Side-by-side 2D + 3D for RF ──────────────────────────────────
fig = plt.figure(figsize=(14, 5.5), facecolor='white')

ax1 = fig.add_subplot(121)
im = ax1.contourf(XX*10, YY*10, T_rf, levels=40, cmap='inferno')
ax1.contour(XX*10, YY*10, T_rf, levels=12, colors='white', linewidths=0.5, alpha=0.6)
cbar = plt.colorbar(im, ax=ax1, shrink=0.9)
cbar.set_label('Temperature (K)', fontsize=9)
ax1.set_xlabel('X (mm)', fontsize=10); ax1.set_ylabel('Y (mm)', fontsize=10)
ax1.set_title('RF-Predicted 2D Thermal Map\n(Hotspot centre at (5,5) mm, σ=0.08)',
              fontsize=10, fontweight='bold')
ax1.set_aspect('equal')
ax1.plot(5, 5, 'w+', markersize=14, markeredgewidth=2, label='Hotspot centre')
ax1.legend(fontsize=8, loc='upper right')

ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(XX*10, YY*10, T_rf, cmap='inferno', edgecolor='none', alpha=0.92)
ax2.set_xlabel('X (mm)', fontsize=8, labelpad=6)
ax2.set_ylabel('Y (mm)', fontsize=8, labelpad=6)
ax2.set_zlabel('T (K)',  fontsize=8, labelpad=6)
ax2.set_title('3D Thermal Surface (RF)', fontsize=10, fontweight='bold')
ax2.view_init(elev=30, azim=-55)
fig.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1).set_label('T (K)', fontsize=8)

plt.suptitle('Spatial Thermal Distribution — RF Model\n'
             '(All other features at training medians; block power = 12 W/mm²)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig_thermal_surface_fixed.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_thermal_surface_fixed.png")

# ── Figure: RF vs MLP 3D side-by-side ────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5), facecolor='white')

for i, (T_grid, label, col) in enumerate([(T_rf, 'Random Forest', 'inferno'),
                                           (T_mlp, 'Neural Net (MLP)', 'plasma')]):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    surf = ax.plot_surface(XX*10, YY*10, T_grid, cmap=col, edgecolor='none', alpha=0.92)
    ax.set_xlabel('X (mm)', fontsize=8, labelpad=6)
    ax.set_ylabel('Y (mm)', fontsize=8, labelpad=6)
    ax.set_zlabel('T (K)',  fontsize=8, labelpad=6)
    ax.set_title(f'3D Thermal Surface\n{label}', fontsize=10, fontweight='bold')
    ax.view_init(elev=30, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1).set_label('T (K)', fontsize=8)

plt.suptitle('RF vs MLP: 3D Predicted Thermal Surface Comparison',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig_3d_comparison.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_3d_comparison.png")

# ── Figure: Temperature profile along cross-section through hotspot ───────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='white')
colors_m = {'OLS': '#4C72B0', 'RF': '#DD8452', 'MLP': '#55A868', 'True': '#C44E52'}

# Load OLS too
ols = LinearRegression()
ols.fit(X_train_sc, y_train)

# Horizontal cross-section at y=0.5 (through hotspot centre)
y_slice = 0.5
x_line  = np.linspace(0, 1, 200)
line_pts = np.tile(med, (200, 1))
line_pts[:, idx['x']]          = x_line
line_pts[:, idx['y']]          = y_slice
line_pts[:, idx['hotspot_cx']] = hcx_fixed
line_pts[:, idx['hotspot_cy']] = hcy_fixed
line_pts[:, idx['sigma_spot']] = sigma_fixed
line_pts[:, idx['block_power_map']] = bpm_fixed
line_pts[:, idx['r_sq']]       = (x_line - hcx_fixed)**2 + (y_slice - hcy_fixed)**2

# True temperature from physics equation
R_si    = med[idx['die_thickness']] / (med[idx['thermal_cond_si']] * 1e-3)
k_eff   = 0.6 * med[idx['metal_cond_Cu']] + 0.4 * med[idx['metal_cond_W']]
R_metal = 0.05 / k_eff
R_tim   = 1.0  / (med[idx['tim_hs_conductance']] * 100)
R_total = R_si + R_metal + R_tim
r2_line = (x_line - hcx_fixed)**2 + (y_slice - hcy_fixed)**2
gaussian = bpm_fixed * np.exp(-r2_line / (2 * sigma_fixed**2))
T_true = med[idx['boundary_temp']] + med[idx['power_density']]*R_total + gaussian*(R_si*0.5 + R_metal)

line_pts_sc = scaler.transform(line_pts)
T_ols_line  = ols.predict(line_pts_sc)
T_rf_line   = rf.predict(line_pts)
T_mlp_line  = mlp.predict(line_pts_sc)

ax = axes[0]
ax.plot(x_line, T_true,     color=colors_m['True'], lw=2.5, label='True (Physics)', zorder=5)
ax.plot(x_line, T_ols_line, color=colors_m['OLS'],  lw=1.8, linestyle='--', label='OLS/Ridge')
ax.plot(x_line, T_rf_line,  color=colors_m['RF'],   lw=1.8, linestyle='-.', label='Random Forest')
ax.plot(x_line, T_mlp_line, color=colors_m['MLP'],  lw=1.8, linestyle=':',  label='MLP')
ax.axvline(hcx_fixed, color='gray', lw=1, linestyle=':', alpha=0.7)
ax.text(hcx_fixed+0.02, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else T_true.min(),
        'Hotspot\ncentre', fontsize=7.5, color='gray')
ax.set_xlabel('X position (normalised)', fontsize=10)
ax.set_ylabel('Temperature (K)', fontsize=10)
ax.set_title('Cross-Section Through Hotspot Centre (y = 0.5)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Vertical cross-section at x=0.5
x_slice = 0.5
y_line  = np.linspace(0, 1, 200)
line_pts2 = np.tile(med, (200, 1))
line_pts2[:, idx['x']]          = x_slice
line_pts2[:, idx['y']]          = y_line
line_pts2[:, idx['hotspot_cx']] = hcx_fixed
line_pts2[:, idx['hotspot_cy']] = hcy_fixed
line_pts2[:, idx['sigma_spot']] = sigma_fixed
line_pts2[:, idx['block_power_map']] = bpm_fixed
line_pts2[:, idx['r_sq']]       = (x_slice - hcx_fixed)**2 + (y_line - hcy_fixed)**2

r2_line2 = (x_slice - hcx_fixed)**2 + (y_line - hcy_fixed)**2
gaussian2 = bpm_fixed * np.exp(-r2_line2 / (2 * sigma_fixed**2))
T_true2 = med[idx['boundary_temp']] + med[idx['power_density']]*R_total + gaussian2*(R_si*0.5 + R_metal)

line_pts2_sc = scaler.transform(line_pts2)
T_ols2  = ols.predict(line_pts2_sc)
T_rf2   = rf.predict(line_pts2)
T_mlp2  = mlp.predict(line_pts2_sc)

ax2 = axes[1]
ax2.plot(y_line, T_true2, color=colors_m['True'], lw=2.5, label='True (Physics)', zorder=5)
ax2.plot(y_line, T_ols2,  color=colors_m['OLS'],  lw=1.8, linestyle='--', label='OLS/Ridge')
ax2.plot(y_line, T_rf2,   color=colors_m['RF'],   lw=1.8, linestyle='-.', label='Random Forest')
ax2.plot(y_line, T_mlp2,  color=colors_m['MLP'],  lw=1.8, linestyle=':',  label='MLP')
ax2.axvline(hcy_fixed, color='gray', lw=1, linestyle=':', alpha=0.7)
ax2.set_xlabel('Y position (normalised)', fontsize=10)
ax2.set_ylabel('Temperature (K)', fontsize=10)
ax2.set_title('Cross-Section Through Hotspot Centre (x = 0.5)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)

plt.suptitle('Temperature Profile Along Chip Cross-Sections — All Models vs True Physics',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/fig_cross_sections.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_cross_sections.png")

print("\nAll fixed figures generated.")


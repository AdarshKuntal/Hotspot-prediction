import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

COLORS = {
    'ols':    '#4C72B0',
    'rf':     '#DD8452',
    'mlp':    '#55A868',
    'accent': '#C44E52',
    'bg':     '#F8F9FA',
}

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('thermal_dataset_v2.csv')

print(f"Dataset shape: {df.shape}")
print(df.describe().round(2))

FEATURES = ['power_density','thermal_cond_si','boundary_temp','metal_cond_Cu',
            'metal_cond_W','die_thickness','block_power_map','tim_hs_conductance',
            'x','y','hotspot_cx','hotspot_cy','sigma_spot','r_sq']
TARGET = 'temperature'

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── Pearson Correlations ─────────────────────────────────────────────────────
corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nPearson correlations with temperature:")
print(corr.round(4))

# ── Models ───────────────────────────────────────────────────────────────────
kf = KFold(5, shuffle=True, random_state=42)

# OLS
ols = LinearRegression()
ols.fit(X_train_sc, y_train)
y_pred_ols = ols.predict(X_test_sc)
cv_ols = cross_val_score(LinearRegression(), X_train_sc, y_train, cv=kf, scoring='r2')

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                           min_samples_split=5, random_state=42,
                           n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
cv_rf = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        X_train, y_train, cv=kf, scoring='r2')
print(f"RF OOB R2: {rf.oob_score_:.4f}")

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                   solver='adam', learning_rate_init=0.001, max_iter=500,
                   early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=10, random_state=42)
mlp.fit(X_train_sc, y_train)
y_pred_mlp = mlp.predict(X_test_sc)
cv_mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500,
                                      early_stopping=True, random_state=42),
                         X_train_sc, y_train, cv=kf, scoring='r2')
print(f"MLP converged at epoch {mlp.n_iter_}")

# ── Metrics ──────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, cv_scores, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    maxe = np.max(np.abs(y_true - y_pred))
    print(f"{label:20s} R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MaxErr={maxe:.2f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    return {'label': label, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MaxErr': maxe,
            'CV_R2': cv_scores.mean(), 'CV_std': cv_scores.std()}

print("\n--- Test Set Results ---")
res_ols   = metrics(y_test, y_pred_ols,   cv_ols,   "OLS Linear Reg.")
res_rf    = metrics(y_test, y_pred_rf,    cv_rf,    "Random Forest")
res_mlp   = metrics(y_test, y_pred_mlp,   cv_mlp,   "Neural Net (MLP)")
results = [res_ols, res_rf, res_mlp]

# ── Feature Importance ───────────────────────────────────────────────────────
fi_df = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_})\
          .sort_values('importance', ascending=False)
print("\nRF Feature Importances:")
print(fi_df.round(4).to_string(index=False))

# ── OLS Coefficients ─────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'feature': FEATURES,
    'OLS': ols.coef_,
}).reindex(pd.Series(np.abs(ols.coef_)).sort_values(ascending=False).index)
print("\nOLS Coefficients (standardised features):")
print(coef_df.round(4).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Chip Cross-Section (recreated)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_facecolor('white')

layers = [
    (0.0, 0.7,  '#8B6914', 'PCB / Motherboard',             'PCB substrate layer'),
    (0.7, 0.5,  '#A0A0A0', 'Solder Bumps / C4',             'Flip-chip solder interconnects'),
    (0.5, 0.7,  '#C8A060', 'Package Substrate',             'PCB-level connection substrate'),
    (0.7, 0.9,  '#6090C8', 'Metal Layers (Back End)',        'Multi-level Cu/W interconnects'),
    (0.9, 1.5,  '#E06060', 'Silicon Die (Active Layer)',     'Transistors, logic — heat source'),
    (1.5, 0.8,  '#5090A0', 'Metal Interconnect (Cu/W)',      'BEOL multi-level routing layers'),
    (0.8, 0.5,  '#90C090', 'TIM Layer 2',                   'Second TIM layer'),
    (0.5, 0.9,  '#D0A050', 'Integrated Heat Spreader (IHS)','Spreads heat across chip'),
    (0.9, 0.6,  '#80B0D0', 'TIM + Thermal Interface',       'Thermal interface material (TIM)'),
    (0.6, 1.0,  '#B0C8E0', 'Heat Sink (Aluminum/Copper)',   'Dissipates heat to ambient air'),
]

y_pos = 0.3
heights = []
y_positions = []
for (_, h, color, name, desc) in layers:
    heights.append(h); y_positions.append(y_pos); y_pos += h + 0.05

# Normalize heights to fill plot
total_h = sum(h + 0.05 for _, h, *_ in layers)
scale = 8.5 / total_h

y_pos = 0.5
for i, (_, h, color, name, desc) in enumerate(layers):
    sh = h * scale
    rect = plt.Rectangle((1.5, y_pos), 6, sh, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(7.65, y_pos + sh/2, name, va='center', fontsize=7.5, fontweight='bold', color='#333')
    ax.text(7.65, y_pos + sh/2 - 0.18, desc, va='center', fontsize=6, color='#666', style='italic')
    y_pos += sh + 0.05 * scale

# Temperature gradient arrows
ax.annotate('', xy=(0.9, 9.0), xytext=(0.9, 0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.35, 5.0, 'HIGH T', fontsize=8, color='red', fontweight='bold', rotation=90, va='center')
ax.annotate('', xy=(0.9, 0.8), xytext=(0.9, 9.0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(0.05, 5.0, 'LOW T', fontsize=8, color='blue', fontweight='bold', rotation=90, va='center')

ax.set_title('Electronic Chip — Cross-Section & Thermal Stack', fontsize=13,
             fontweight='bold', pad=10, color='#222')

plt.tight_layout()
plt.savefig('fig_cross_section.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_cross_section.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Chip Top View (Block power map + RF thermal heatmap)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')

# Left: block power map
ax1 = axes[0]
blocks = [
    dict(x=0.0, y=6.0, w=4.5, h=4.0, power=11.5, label='CPU\nCore 0',  color='#E05050'),
    dict(x=4.5, y=6.0, w=3.5, h=4.0, power=12.8, label='CPU\nCore 1',  color='#E05050'),
    dict(x=8.0, y=6.0, w=2.0, h=4.0, power=6.5,  label='VRM',         color='#E08030'),
    dict(x=0.0, y=3.5, w=4.5, h=2.5, power=9.2,  label='GPU/Compute', color='#D06040'),
    dict(x=4.5, y=3.5, w=3.5, h=2.5, power=4.1,  label='Cache (L3)',  color='#70A870'),
    dict(x=8.0, y=3.5, w=2.0, h=2.5, power=4.5,  label='I/O',         color='#6080C0'),
    dict(x=0.0, y=0.0, w=6.5, h=3.5, power=0.9,  label='DRAM / Memory', color='#5090B0'),
    dict(x=6.5, y=0.0, w=3.5, h=3.5, power=1.2,  label='PMU',         color='#8070B0'),
]
for b in blocks:
    rect = plt.Rectangle((b['x'], b['y']), b['w'], b['h'],
                          facecolor=b['color'], edgecolor='white', linewidth=2, alpha=0.85)
    ax1.add_patch(rect)
    ax1.text(b['x'] + b['w']/2, b['y'] + b['h']/2 + 0.3, b['label'],
             ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')
    ax1.text(b['x'] + b['w']/2, b['y'] + b['h']/2 - 0.45,
             f"{b['power']} W/mm²",
             ha='center', va='center', fontsize=6.5, color='#FFFFFFCC')
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.set_xlabel('X (mm)', fontsize=10); ax1.set_ylabel('Y (mm)', fontsize=10)
ax1.set_title('Chip Top View — Block Power Map', fontsize=11, fontweight='bold')
ax1.set_aspect('equal')

# Right: RF-predicted thermal map
ax2 = axes[1]
res_grid = 80
xg = np.linspace(0, 1, res_grid)
yg = np.linspace(0, 1, res_grid)
XX, YY = np.meshgrid(xg, yg)
med = np.median(X_train, axis=0)
grid_pts = np.tile(med, (res_grid*res_grid, 1))
xi = FEATURES.index('x'); yi_idx = FEATURES.index('y')
grid_pts[:, xi] = XX.ravel()
grid_pts[:, yi_idx] = YY.ravel()
# update r_sq
hcx = FEATURES.index('hotspot_cx'); hcy = FEATURES.index('hotspot_cy')
rsq_idx = FEATURES.index('r_sq')
grid_pts[:, rsq_idx] = (grid_pts[:,xi] - grid_pts[:,hcx])**2 + (grid_pts[:,yi_idx] - grid_pts[:,hcy])**2
T_pred = rf.predict(grid_pts).reshape(res_grid, res_grid)

# Scale to mm
im = ax2.contourf(XX*10, YY*10, T_pred, levels=30, cmap='inferno')
ax2.contour(XX*10, YY*10, T_pred, levels=10, colors='white', linewidths=0.4, alpha=0.5)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Temperature (K)', fontsize=9)
ax2.set_xlabel('X (mm)', fontsize=10); ax2.set_ylabel('Y (mm)', fontsize=10)
ax2.set_title('Predicted Thermal Map (RF Model)', fontsize=11, fontweight='bold')
ax2.set_aspect('equal')

plt.suptitle('Chip Layout & Thermal Distribution', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_chip_topview.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_chip_topview.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Actual vs Predicted (all 3 models)
# ════════════════════════════════════════════════════════════════════════════
model_data = [
    ('OLS', y_pred_ols, COLORS['ols'],   res_ols),
    ('Random Forest', y_pred_rf,  COLORS['rf'],    res_rf),
    ('Neural Net (MLP)', y_pred_mlp, COLORS['mlp'], res_mlp),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='white')
for ax, (name, y_pred, col, res) in zip(axes, model_data):
    ax.scatter(y_test, y_pred, c=col, alpha=0.25, s=8, rasterized=True)
    lims = [min(y_test.min(), y_pred.min())-2, max(y_test.max(), y_pred.max())+2]
    ax.plot(lims, lims, 'k--', lw=1.2, label='Ideal (1:1)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual Temperature (K)', fontsize=9)
    ax.set_ylabel('Predicted Temperature (K)', fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold', color=col)
    ax.text(0.05, 0.92, f"R²={res['R2']:.4f}\nRMSE={res['RMSE']:.3f} K",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.8))
    ax.set_aspect('equal')
plt.suptitle('Actual vs. Predicted Temperature — Test Set (n=2,400)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_actual_vs_pred.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_actual_vs_pred.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Residual Distributions
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
for ax, (name, y_pred, col, res) in zip(axes, model_data):
    resid = y_test - y_pred
    ax.hist(resid, bins=50, color=col, alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axvline(0, color='black', lw=1.2, linestyle='--')
    ax.axvline(resid.mean(), color=COLORS['accent'], lw=1.2, linestyle=':')
    ax.set_xlabel('Residual (K)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold', color=col)
    ax.text(0.97, 0.95, f"μ={resid.mean():.3f} K\nσ={resid.std():.3f} K",
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.8))
plt.suptitle('Residual Distributions — Test Set', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_residuals.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_residuals.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Performance Bar Chart (R², RMSE, MAE)
# ════════════════════════════════════════════════════════════════════════════
labels = ['OLS', 'Random Forest', 'Neural Net (MLP)']
r2s   = [res_ols['R2'],   res_rf['R2'],   res_mlp['R2']]
rmses = [res_ols['RMSE'], res_rf['RMSE'], res_mlp['RMSE']]
maes  = [res_ols['MAE'],  res_rf['MAE'],  res_mlp['MAE']]
cols  = [COLORS['ols'],   COLORS['rf'],   COLORS['mlp']]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor='white')
x = np.arange(len(labels))
for ax, vals, ylabel, title, better in zip(
        axes,
        [r2s, rmses, maes],
        ['R²', 'RMSE (K)', 'MAE (K)'],
        ['R² (higher → better)', 'RMSE (lower → better)', 'MAE (lower → better)'],
        ['higher', 'lower', 'lower']):
    bars = ax.bar(x, vals, color=cols, edgecolor='white', linewidth=1.2, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003*max(vals),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, rotation=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    if better == 'higher':
        ax.set_ylim(0.85, 1.02)
    else:
        ax.set_ylim(0, max(vals)*1.25)
plt.suptitle('Model Performance Comparison — 14-Feature Dataset',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_performance_bars.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_performance_bars.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Feature Importance Triptych (RF MDI + OLS coef + Pearson)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='white')

# RF MDI
ax = axes[0]
fi_plot = fi_df.head(10)
bars = ax.barh(fi_plot['feature'][::-1], fi_plot['importance'][::-1],
               color=COLORS['rf'], alpha=0.85, edgecolor='white')
ax.set_xlabel('MDI Score', fontsize=9)
ax.set_title('RF Feature Importance (MDI)', fontsize=10, fontweight='bold')
for bar, val in zip(bars, fi_plot['importance'][::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7.5)

# OLS |coefs|
ax = axes[1]
coef_sorted = pd.DataFrame({'feature': FEATURES, 'coef': np.abs(ols.coef_)})\
                .sort_values('coef', ascending=True).tail(10)
ax.barh(coef_sorted['feature'], coef_sorted['coef'],
        color=COLORS['ols'], alpha=0.85, edgecolor='white')
ax.set_xlabel('|Coefficient| (standardised)', fontsize=9)
ax.set_title('OLS |Coefficients|', fontsize=10, fontweight='bold')

# Pearson
ax = axes[2]
corr_sorted = corr.sort_values()
bar_colors = [COLORS['accent'] if v < 0 else COLORS['ols'] for v in corr_sorted]
ax.barh(corr_sorted.index, corr_sorted.values, color=bar_colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Pearson r with Temperature', fontsize=9)
ax.set_title('Pearson Correlation', fontsize=10, fontweight='bold')
neg_patch = mpatches.Patch(color=COLORS['accent'], alpha=0.85, label='Negative')
pos_patch = mpatches.Patch(color=COLORS['ols'],    alpha=0.85, label='Positive')
ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc='lower right')

plt.suptitle('Feature Analysis: RF Importance · OLS Coefficients · Pearson Correlation',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_feature_analysis.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_feature_analysis.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — MLP Training Loss Curve
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
ax.plot(mlp.loss_curve_, color=COLORS['mlp'], lw=2, label='Training Loss (MSE)')
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ is not None:
    val_loss = [-s for s in mlp.validation_scores_]
    ax.plot(val_loss, color=COLORS['accent'], lw=1.5, linestyle='--', label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('MSE Loss', fontsize=10)
ax.set_title('MLP Training Loss Curve', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig_mlp_loss.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_mlp_loss.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Error vs r_sq  (OLS and RF)
# ════════════════════════════════════════════════════════════════════════════
rsq_test = X_test[:, FEATURES.index('r_sq')]
err_ols  = np.abs(y_test - y_pred_ols)
err_rf   = np.abs(y_test - y_pred_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='white')
for ax, err, col, name in zip(axes,
                               [err_ols, err_rf],
                               [COLORS['ols'], COLORS['rf']],
                               ['OLS: Error vs Hotspot Distance',
                                'RF: Error vs Hotspot Distance']):
    ax.scatter(rsq_test, err, c=col, alpha=0.2, s=6, rasterized=True)
    ax.set_xlabel('r_sq (distance² from hotspot centre)', fontsize=9)
    ax.set_ylabel('Absolute Error (K)', fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold', color=col)
plt.suptitle('Prediction Error vs. Spatial Distance from Hotspot Centre',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_error_vs_rsq.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_error_vs_rsq.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Feature Correlation Matrix
# ════════════════════════════════════════════════════════════════════════════
corr_mat = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 7}, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (14 Features)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_corr_matrix.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_corr_matrix.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURES 11–13 — Thermal Surface Maps (fixed hotspot parameters)
# ════════════════════════════════════════════════════════════════════════════
from mpl_toolkits.mplot3d import Axes3D

# Build a high-resolution grid with physically meaningful hotspot parameters
res_fixed = 60
xg_f = np.linspace(0, 1, res_fixed)
yg_f = np.linspace(0, 1, res_fixed)
XX_f, YY_f = np.meshgrid(xg_f, yg_f)

med_f = np.median(X_train, axis=0).copy()

hcx_fixed   = 0.5
hcy_fixed   = 0.5
sigma_fixed = 0.08   # sharp hotspot for clear peak
bpm_fixed   = 12.0   # high block power to amplify hotspot

idx = {f: i for i, f in enumerate(FEATURES)}

grid_pts_f = np.tile(med_f, (res_fixed * res_fixed, 1))
grid_pts_f[:, idx['x']]               = XX_f.ravel()
grid_pts_f[:, idx['y']]               = YY_f.ravel()
grid_pts_f[:, idx['hotspot_cx']]      = hcx_fixed
grid_pts_f[:, idx['hotspot_cy']]      = hcy_fixed
grid_pts_f[:, idx['sigma_spot']]      = sigma_fixed
grid_pts_f[:, idx['block_power_map']] = bpm_fixed
grid_pts_f[:, idx['r_sq']]            = (XX_f.ravel() - hcx_fixed)**2 + (YY_f.ravel() - hcy_fixed)**2

T_rf_f  = rf.predict(grid_pts_f).reshape(res_fixed, res_fixed)
grid_pts_f_sc = scaler.transform(grid_pts_f)
T_mlp_f = mlp.predict(grid_pts_f_sc).reshape(res_fixed, res_fixed)

print(f"\nFixed-hotspot RF  T range: {T_rf_f.min():.2f} -- {T_rf_f.max():.2f} K")
print(f"Fixed-hotspot MLP T range: {T_mlp_f.min():.2f} -- {T_mlp_f.max():.2f} K")

# ── FIGURE 11 — Side-by-side 2D + 3D for RF (fixed hotspot) ─────────────
fig = plt.figure(figsize=(14, 5.5), facecolor='white')

ax1 = fig.add_subplot(121)
im = ax1.contourf(XX_f*10, YY_f*10, T_rf_f, levels=40, cmap='inferno')
ax1.contour(XX_f*10, YY_f*10, T_rf_f, levels=12, colors='white', linewidths=0.5, alpha=0.6)
cbar = plt.colorbar(im, ax=ax1, shrink=0.9)
cbar.set_label('Temperature (K)', fontsize=9)
ax1.set_xlabel('X (mm)', fontsize=10); ax1.set_ylabel('Y (mm)', fontsize=10)
ax1.set_title('RF-Predicted 2D Thermal Map\n(Hotspot centre at (5,5) mm, σ=0.08)',
              fontsize=10, fontweight='bold')
ax1.set_aspect('equal')
ax1.plot(5, 5, 'w+', markersize=14, markeredgewidth=2, label='Hotspot centre')
ax1.legend(fontsize=8, loc='upper right')

ax2 = fig.add_subplot(122, projection='3d')
surf = ax2.plot_surface(XX_f*10, YY_f*10, T_rf_f, cmap='inferno', edgecolor='none', alpha=0.92)
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
plt.savefig('fig_thermal_surface_fixed.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_thermal_surface_fixed.png")

# ── FIGURE 12 — RF vs MLP 3D side-by-side ───────────────────────────────
fig = plt.figure(figsize=(14, 5.5), facecolor='white')

for i, (T_grid, label, col) in enumerate([(T_rf_f,  'Random Forest',    'inferno'),
                                           (T_mlp_f, 'Neural Net (MLP)', 'plasma')]):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    surf = ax.plot_surface(XX_f*10, YY_f*10, T_grid, cmap=col, edgecolor='none', alpha=0.92)
    ax.set_xlabel('X (mm)', fontsize=8, labelpad=6)
    ax.set_ylabel('Y (mm)', fontsize=8, labelpad=6)
    ax.set_zlabel('T (K)',  fontsize=8, labelpad=6)
    ax.set_title(f'3D Thermal Surface\n{label}', fontsize=10, fontweight='bold')
    ax.view_init(elev=30, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1).set_label('T (K)', fontsize=8)

plt.suptitle('RF vs MLP: 3D Predicted Thermal Surface Comparison',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_3d_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_3d_comparison.png")

# ── FIGURE 13 — Temperature profile along cross-sections through hotspot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='white')
colors_m = {'OLS': '#4C72B0', 'RF': '#DD8452', 'MLP': '#55A868', 'True': '#C44E52'}

# Horizontal cross-section at y=0.5 (through hotspot centre)
y_slice = 0.5
x_line  = np.linspace(0, 1, 200)
line_pts = np.tile(med_f, (200, 1))
line_pts[:, idx['x']]               = x_line
line_pts[:, idx['y']]               = y_slice
line_pts[:, idx['hotspot_cx']]      = hcx_fixed
line_pts[:, idx['hotspot_cy']]      = hcy_fixed
line_pts[:, idx['sigma_spot']]      = sigma_fixed
line_pts[:, idx['block_power_map']] = bpm_fixed
line_pts[:, idx['r_sq']]            = (x_line - hcx_fixed)**2 + (y_slice - hcy_fixed)**2

# True temperature from physics equation
R_si    = med_f[idx['die_thickness']] / (med_f[idx['thermal_cond_si']] * 1e-3)
k_eff   = 0.6 * med_f[idx['metal_cond_Cu']] + 0.4 * med_f[idx['metal_cond_W']]
R_metal = 0.05 / k_eff
R_tim   = 1.0  / (med_f[idx['tim_hs_conductance']] * 100)
R_total = R_si + R_metal + R_tim
r2_line   = (x_line - hcx_fixed)**2 + (y_slice - hcy_fixed)**2
gaussian  = bpm_fixed * np.exp(-r2_line / (2 * sigma_fixed**2))
T_true    = med_f[idx['boundary_temp']] + med_f[idx['power_density']]*R_total + gaussian*(R_si*0.5 + R_metal)

line_pts_sc = scaler.transform(line_pts)
T_ols_line  = ols.predict(line_pts_sc)
T_rf_line   = rf.predict(line_pts)
T_mlp_line  = mlp.predict(line_pts_sc)

ax = axes[0]
ax.plot(x_line, T_true,     color=colors_m['True'], lw=2.5, label='True (Physics)', zorder=5)
ax.plot(x_line, T_ols_line, color=colors_m['OLS'],  lw=1.8, linestyle='--', label='OLS')
ax.plot(x_line, T_rf_line,  color=colors_m['RF'],   lw=1.8, linestyle='-.', label='Random Forest')
ax.plot(x_line, T_mlp_line, color=colors_m['MLP'],  lw=1.8, linestyle=':',  label='MLP')
ax.axvline(hcx_fixed, color='gray', lw=1, linestyle=':', alpha=0.7)
ax.text(hcx_fixed + 0.02, T_true.min(), 'Hotspot\ncentre', fontsize=7.5, color='gray')
ax.set_xlabel('X position (normalised)', fontsize=10)
ax.set_ylabel('Temperature (K)', fontsize=10)
ax.set_title('Cross-Section Through Hotspot Centre (y = 0.5)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Vertical cross-section at x=0.5
x_slice = 0.5
y_line  = np.linspace(0, 1, 200)
line_pts2 = np.tile(med_f, (200, 1))
line_pts2[:, idx['x']]               = x_slice
line_pts2[:, idx['y']]               = y_line
line_pts2[:, idx['hotspot_cx']]      = hcx_fixed
line_pts2[:, idx['hotspot_cy']]      = hcy_fixed
line_pts2[:, idx['sigma_spot']]      = sigma_fixed
line_pts2[:, idx['block_power_map']] = bpm_fixed
line_pts2[:, idx['r_sq']]            = (x_slice - hcx_fixed)**2 + (y_line - hcy_fixed)**2

r2_line2  = (x_slice - hcx_fixed)**2 + (y_line - hcy_fixed)**2
gaussian2 = bpm_fixed * np.exp(-r2_line2 / (2 * sigma_fixed**2))
T_true2   = med_f[idx['boundary_temp']] + med_f[idx['power_density']]*R_total + gaussian2*(R_si*0.5 + R_metal)

line_pts2_sc = scaler.transform(line_pts2)
T_ols2  = ols.predict(line_pts2_sc)
T_rf2   = rf.predict(line_pts2)
T_mlp2  = mlp.predict(line_pts2_sc)

ax2 = axes[1]
ax2.plot(y_line, T_true2, color=colors_m['True'], lw=2.5, label='True (Physics)', zorder=5)
ax2.plot(y_line, T_ols2,  color=colors_m['OLS'],  lw=1.8, linestyle='--', label='OLS')
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
plt.savefig('fig_cross_sections.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_cross_sections.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Heat Dissipation Pathway (schematic)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='white')
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis('off')

boxes_top = [
    (0.2, 2.6, 2.1, 1.8, '#4C72B0', 'Electrical\nPower Input',  'V·I = P (Watts)'),
    (2.7, 2.6, 2.1, 1.8, '#DD8452', 'Transistor\nSwitching',    'Power dissipation'),
    (5.2, 2.6, 2.1, 1.8, '#C44E52', 'Si Die\n(Heat Source)',    'Joule heating\nq⃗ = −k∇T'),
    (7.7, 2.6, 2.1, 1.8, '#8B4513', 'Hotspot\nFormation',       'T peaks at high density'),
]
boxes_bot = [
    (7.7, 0.6, 2.1, 1.8, '#5090B0', 'Heat Sink',                'Convection to air'),
    (5.2, 0.6, 2.1, 1.8, '#70A870', 'TIM + IHS',                'Interface resistance'),
    (2.7, 0.6, 2.1, 1.8, '#6B4C9A', 'Metal Layers\n(Cu/W)',     'Conduction k∇T'),
    (0.2, 0.6, 2.1, 1.8, '#C44E52', 'Thermal\nThrottling Risk', 'Perf ↓ if T > Tmax'),
]
for bx, by, bw, bh, bc, bn, bd in boxes_top + boxes_bot:
    rect = plt.Rectangle((bx, by), bw, bh, facecolor=bc, edgecolor='white',
                          linewidth=2, alpha=0.88, zorder=2)
    ax.add_patch(rect)
    ax.text(bx+bw/2, by+bh/2+0.22, bn, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=3)
    ax.text(bx+bw/2, by+bh/2-0.32, bd, ha='center', va='center',
            fontsize=6.5, color='#FFFFFF99', zorder=3)

arrow_kw = dict(arrowstyle='->', color='#444', lw=1.8, connectionstyle='arc3,rad=0')
for (x1, x2, y) in [(2.3,2.7,3.5),(4.8,5.2,3.5),(7.3,7.7,3.5)]:
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.8))
ax.annotate('', xy=(8.75, 2.6), xytext=(8.75, 2.4),
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.8))
for (x1, x2, y) in [(7.3,6.5,1.5),(4.8,4.2,1.5),(2.3,1.7,1.5)]:
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.8))

ax.text(5.5, 4.7, 'Heat Generation & Dissipation Pathway in Electronic Chips',
        ha='center', fontsize=11, fontweight='bold', color='#222')
ax.text(5.5, 4.3, 'Electrical power → Joule heating in Si die → Conduction through metal layers → TIM → Heat sink → Ambient',
        ha='center', fontsize=7.5, color='#555', style='italic')

plt.tight_layout()
plt.savefig('fig_heat_pathway.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_heat_pathway.png")

print("\n=== ALL FIGURES SAVED ===")
print(" 1. fig_cross_section.png       — Chip layer cross-section")
print(" 2. fig_chip_topview.png        — Block power map + RF thermal heatmap")
print(" 3. fig_actual_vs_pred.png      — Actual vs Predicted (all models)")
print(" 4. fig_residuals.png           — Residual distributions")
print(" 5. fig_performance_bars.png    — R², RMSE, MAE comparison")
print(" 6. fig_feature_analysis.png    — RF MDI + OLS coef + Pearson")
print(" 8. fig_mlp_loss.png            — MLP training loss curve")
print(" 9. fig_error_vs_rsq.png        — Error vs hotspot distance")
print("10. fig_corr_matrix.png         — Feature correlation matrix")
print("11. fig_thermal_surface_fixed.png — 2D + 3D RF thermal map (fixed hotspot)")
print("12. fig_3d_comparison.png       — RF vs MLP 3D surface comparison")
print("13. fig_cross_sections.png      — Cross-section profiles vs true physics")
print("14. fig_heat_pathway.png        — Heat dissipation pathway schematic")

print(f"\nSummary Table:")
print(f"{'Model':<22} {'R2':>7} {'RMSE':>8} {'MAE':>8} {'MaxErr':>9} {'CV R2':>12}")
print("-"*68)
for r in results:
    print(f"{r['label']:<22} {r['R2']:>7.4f} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} {r['MaxErr']:>9.2f} {r['CV_R2']:>7.4f}±{r['CV_std']:.4f}")

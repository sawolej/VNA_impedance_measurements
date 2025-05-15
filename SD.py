# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar

# 1) Dane kalibracyjne: {nazwa: (Z_meas [Ω], ε_r_tab)}
data = {
    'air':        (303.50,   1.00),
    'pla':        (279.17,   2.70),
    'cooper-pla': (223.07,   3.50),
    'alko':       ( 89.73,  30.00),
    'woda':       ( 41.75,  80.10),
}

materials = list(data.keys())
Z_meas  = np.array([data[m][0] for m in materials])
eps_tab = np.array([data[m][1] for m in materials])

# 2) Budujemy interpolator PCHIP
f_interp = PchipInterpolator(eps_tab, Z_meas, extrapolate=True)

# 3) Gęsta siatka do wykresu
eps_lin = np.linspace(eps_tab.min(), eps_tab.max(), 500)
Z_lin   = f_interp(eps_lin)

# 4) Wartość nieznana
Z_unk = 280.0
diff = Z_lin - Z_unk
idxs = np.where(diff[:-1] * diff[1:] < 0)[0]
if len(idxs) > 0:
    i = idxs[0]
    a, b = eps_lin[i], eps_lin[i+1]
    sol = root_scalar(lambda e: f_interp(e) - Z_unk,
                      bracket=[a, b], method='brentq')
    eps_est = sol.root if sol.converged else np.nan
else:
    i_min = np.argmin(np.abs(diff))
    eps_est = eps_lin[i_min]

print(f"Oszacowane ε_r dla Z_unk={Z_unk} Ω: {eps_est:.3f}")

# 5) Rysowanie
plt.figure(figsize=(6,5))
# punkty pomiarowe
plt.scatter(eps_tab, Z_meas, s=80, label='punkty pomiarowe')
# krzywa interp.
plt.plot(eps_lin, Z_lin, '-', label='PCHIP spline')
# punkt nieznany
plt.scatter(eps_est, Z_unk, color='red', s=100, marker='o', label=f'Z_unk={Z_unk} Ω\nε_r≈{eps_est:.2f}')
plt.xlabel(r'$\varepsilon_r$')
plt.ylabel(r'$Z$ [Ω]')
plt.title('Impedancja vs. stała dielektryczna')
plt.legend()
plt.grid(True, ls='--', lw=0.5)
plt.tight_layout()
plt.show()

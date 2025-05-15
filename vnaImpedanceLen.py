# -*- coding: utf-8 -*-
"""
Time-gating + de-embedding |Zin| – zapis wszystkich wykresów
wersja 2025-05-15 — AUTO-fit fazy lub RT_DELAY = 4.2 ns
"""

from __future__ import annotations
import io, re
import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import skrf as rf      # pip install scikit-rf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PLIK I PARAMETRY
# ---------------------------------------------------------------------
FILES = [
    # woda
    r"C:/Users/kamil/Documents/timeGating/woda_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/woda_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/woda-siglentVNA2025-05-15-01-56-05.s1p",
    # pla
    r"C:/Users/kamil/Documents/timeGating/pla_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pla_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/pla-siglentVNA2025-05-15-01-53-13.s1p",
    # metal
    r"C:/Users/kamil/Documents/timeGating/metal_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/metal_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/metal-siglentVNA2025-05-15-01-54-01.s1p",
    # cooper
    r"C:/Users/kamil/Documents/timeGating/cooper-pla_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/cooper-pla_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/cooper-pla_siglentVNA2025-05-15-01-54-42.s1p",
    # alko
    r"C:/Users/kamil/Documents/timeGating/alko_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/alko_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/alko-siglentVNA2025-05-15-01-55-18.s1p",
    # air
    r"C:/Users/kamil/Documents/timeGating/air_megiq-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/air_pico-S11-VF.s1p",
    r"C:/Users/kamil/Documents/timeGating/air-siglentVNA2025-05-15-01-52-32.s1p",
]


GATE_CENTER_NS       = 5.0
GATE_SPAN_NS         = 0.5
Z0                   = 50.0
AUTO_FIT_DELAY       = False     # True → fit fazy; False → stały
CABLE_LENGTH_M       = 0.51      # [m], 0.50 dla MEGIQ
SIGNAL_DELAY_NS_PER_M = 4.86     # [ns/m]
TARGET_GHZ           = [1.223, 1.333, 1.600, 2.400]
SAMPLE_ORDER         = ["air","pla","cooper-pla","alko","woda","metal"]
INSTR_ORDER          = ["MEGIQ","PICO","SIGLENT"]

# ---------------------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------------------
def load_network_any_decimal(path: Union[str, Path]) -> rf.Network:
    txt   = Path(path).read_text(encoding="utf-8")
    fixed = re.sub(r"(?<=\d),(?=\d)", ".", txt)
    buf   = io.StringIO(fixed); buf.name = str(path)
    return rf.Network(buf)

def sample_and_instr(fname: str) -> tuple[str,str]:
    s = Path(fname).stem.lower()
    sample = next((x for x in SAMPLE_ORDER if s.startswith(x)), None)
    instr  = next((x for x in INSTR_ORDER if x.lower() in s), None)
    if sample is None or instr is None:
        raise ValueError(f"Nieznany sample/instr w '{fname}'")
    return sample, instr

def estimate_rt_delay(gamma: np.ndarray, freqs: np.ndarray) -> float:
    phase = np.unwrap(np.angle(gamma))
    slope, _ = np.polyfit(freqs, phase, 1)
    return -slope / (2*np.pi)

# ---------------------------------------------------------------------
# KATALOG WYJŚCIOWY
# ---------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("TIMEGATING%Y%m%d_%H%M%S")
output_dir = Path.cwd() / timestamp
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Zapisuję wszystkie wykresy w: {output_dir}")

# ---------------------------------------------------------------------
# PRZETWARZANIE I ZAPIS WYKRESÓW S11
# ---------------------------------------------------------------------
rf.stylely()
results = {}
spectra = {}

for path in FILES:
    net   = load_network_any_decimal(path)
    s11   = net.s11
    s11_g = s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS)
    stem  = Path(path).stem

    # 1) S₁₁ raw vs gated – jedna figura 2x1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s11.plot_s_db(ax=axes[0], label="raw")
    s11_g.plot_s_db(ax=axes[0], label="gated")
    axes[0].set_title(f"{stem} – S11 dB (f)")
    axes[0].legend()
    s11.plot_s_db_time(ax=axes[1], label="raw")
    s11_g.plot_s_db_time(ax=axes[1], label="gated")
    axes[1].set_xlim(0, 40)
    axes[1].set_title(f"{stem} – S11 dB (t)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_S11_raw_gated.png")
    plt.close(fig)

    # 2) De-embedding → |Zin|
    f     = s11_g.f
    gamma = s11_g.s.flatten()
    sample, instr = sample_and_instr(stem)

    if AUTO_FIT_DELAY:
        RT_DELAY = estimate_rt_delay(gamma, f)
    else:
        length_m = 0.50 if instr=="MEGIQ" else CABLE_LENGTH_M
        tof_one_way = SIGNAL_DELAY_NS_PER_M * length_m * 1e-9
        RT_DELAY    = tof_one_way

    gamma_tip = gamma * np.exp(1j * 2*np.pi * f * RT_DELAY)
    zin = Z0 * (1 + gamma_tip) / (1 - gamma_tip)
    zin_abs = np.abs(zin)

    spectra[(sample, instr)] = (f/1e9, zin_abs)
    for tg in TARGET_GHZ:
        idx = np.argmin(np.abs(f/1e9 - tg))
        results[(sample, instr, f"{tg:.3f} GHz")] = zin_abs[idx]

# ---------------------------------------------------------------------
# TABELKI |Zin|
# ---------------------------------------------------------------------
freq_cols = [f"{tg:.3f} GHz" for tg in TARGET_GHZ]
for instr in INSTR_ORDER:
    df = pd.DataFrame(index=SAMPLE_ORDER, columns=freq_cols, dtype=float)
    for (samp, inst, fstr), val in results.items():
        if inst == instr:
            df.loc[samp, fstr] = val
    print(f"\n==== Instrument: {instr} ====")
    with pd.option_context("display.width", None, "display.max_columns", None):
        print(df.round(2))

# ---------------------------------------------------------------------
# ZAPIS WYKRESÓW |Zin| vs częstotliwość
# ---------------------------------------------------------------------
for instr in INSTR_ORDER:
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    for samp in SAMPLE_ORDER:
        key = (samp, instr)
        if key in spectra:
            freqs_ghz, zin_abs = spectra[key]
            ax.plot(freqs_ghz, zin_abs, label=samp)
    ax.set_title(f"|Zin| – {instr}")
    ax.set_xlabel("Częstotliwość [GHz]")
    ax.set_ylabel("|Zin| [Ω]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{instr}_Zin_vs_freq.png")
    plt.show()
    plt.close(fig)

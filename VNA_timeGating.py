# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path

import skrf as rf
import matplotlib.pyplot as plt


files = [
    r'C:/Users/kamil/Documents/timeGating/pico108.s1p',  # air
    r'C:/Users/kamil/Documents/timeGating/pico106.s1p',  # PLA
    r'C:/Users/kamil/Documents/timeGating/megiq.s1p',  # metal
     r'C:/Users/kamil/Documents/timeGating/siglent.s1p',  # metal
]
MAX_F_GHZ = 4      # górna granica pasma
F_SLICE   = f'-{MAX_F_GHZ}ghz'   

def load_touchstone_with_commas(path: str) -> rf.Network:
    """Wczytuje plik .sNp zapisany z przecinkami dziesiętnymi."""
    with open(path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    fixed_lines = []
    for ln in lines:
        # Komentarze (!) i linia formatu (#) zostawiamy nietknięte
        if ln.lstrip().startswith(("!", "#")):
            fixed_lines.append(ln)
        else:
            fixed_lines.append(ln.replace(",", "."))

    tmp = tempfile.NamedTemporaryFile(
        "w+", delete=False, suffix=Path(path).suffix, encoding="utf-8"
    )
    tmp.writelines(fixed_lines)
    tmp.flush()
    tmp.close()

    return rf.Network(tmp.name)



GATE_CENTER_NS = 5     # środek okna (ns)
GATE_SPAN_NS   = 2   
networks, gated = [], []
for p in files:
    ntwk = load_touchstone_with_commas(p)

    # ntwk = ntwk[F_SLICE] 
    networks.append(ntwk)
    gated.append(ntwk.s11.time_gate(center=GATE_CENTER_NS, span=GATE_SPAN_NS,
                             ))


plt.figure(figsize=(10, 9))

for row, (ntwk, ntwk_gated, path) in enumerate(zip(networks, gated, files), start=1):

    # Kolumna 1 – |S11| w dB (dziedzina f)
    ax_f = plt.subplot(len(files), 2, 2*row - 1)
    ntwk.s11.plot_s_db(ax=ax_f, label="raw")
    ntwk_gated.plot_s_db(ax=ax_f, label="gated")
    ax_f.set_title(f"{Path(path).name}  –  Frequency")
    # ax_f.set_ylim(-4, 0) 
    ax_f.legend()

    # Kolumna 2 – |S11| w dB (dziedzina t)
    ax_t = plt.subplot(len(files), 2, 2*row)
    ntwk.s11.plot_s_db_time(ax=ax_t, label="raw")
    ntwk_gated.plot_s_db_time(ax=ax_t, label="gated")
    ax_t.set_xlim(0, 40)          # dostosuj do potrzeb
    ax_t.set_title(f"{Path(path).name}  –  Time")
    ax_t.legend()

plt.tight_layout()
plt.show()

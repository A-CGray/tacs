"""
==============================================================================

==============================================================================
@File    :   plotEquilibriumPath.py
@Date    :   2024/10/07
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import pickle

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import niceplots

# ==============================================================================
# Extension modules
# ==============================================================================

plt.style.use(niceplots.get_style())
niceColors = niceplots.get_colors()

# ==============================================================================
# Load data
# ==============================================================================
# Reference data taken from Table 9 (d) in "Popular benchmark problems for geometric
# nonlinear analysis of shells" by Sze et al, (doi:10.1016/j.!nel.2003.11.001)
refDataArray = np.loadtxt("Sze-Data.csv", skiprows=1, delimiter=",")
refData = {"loadScale": refDataArray[:, 0], "zDisp": refDataArray[:, 1]}
# Interpolate the reference data so we can plot it as a smooth line
refDataInterp = {}
for key, value in refData.items():
    interpolator = sp.interpolate.interp1d(
        np.linspace(0, 1, len(value)), value, kind="cubic"
    )
    refDataInterp[key] = interpolator(np.linspace(0, 1, 10 * len(value)))

with open("TACS-Disps-nonlinear_quadratic_load-Incrementation.pkl", "rb") as f:
    loadIncData = pickle.load(f)
    loadIncData["zDisp"] *= -1e3

with open("TACS-Disps-nonlinear_quadratic_arcLength-Incrementation.pkl", "rb") as f:
    arcLengthData = pickle.load(f)
    arcLengthData["zDisp"] *= -1e3

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel("Vertical Displacement (mm)")
ax.set_ylabel("Load Scale")

ax.plot(
    "zDisp",
    "loadScale",
    data=refDataInterp,
    label="Abaqus (Sze et al)",
    color=niceColors["Axis"],
    clip_on=False,
)
# ax.plot("zDisp", "loadScale", data=refData, linestyle="", marker="o", color=niceColors["Axis"], clip_on=False, label=None)

ax.plot(
    "zDisp",
    "loadScale",
    data=loadIncData,
    linestyle="",
    marker="o",
    markersize=6,
    label="TACS: Load Incrementation",
    color=niceColors["Yellow"],
    clip_on=False,
)
ax.plot(
    "zDisp",
    "loadScale",
    data=arcLengthData,
    linestyle="",
    marker="o",
    markersize=6,
    label="TACS: Arc-length Incrementation",
    color=niceColors["Blue"],
    clip_on=False,
)

niceplots.adjust_spines(ax)
ax.legend(labelcolor="linecolor")

niceplots.save_figs(fig, "EquilibriumPath", ["pdf", "png"])

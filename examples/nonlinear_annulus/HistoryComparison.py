"""
==============================================================================

==============================================================================
@File    :   HistoryComparison.py
@Date    :   2023/06/09
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import pickle
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import matplotlib.pyplot as plt
import niceplots
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
plt.style.use(niceplots.get_style())
fig, ax = plt.subplots()

cases = ["WithoutPredictor", "WithPredictor"]
caseNames = ["Without predictor", "With predictor"]

for case, caseName in zip(cases, caseNames):
    with open(os.path.join(case, f"Annulus_000.pkl"), "rb") as f:
        history = pickle.load(f)
    ax.plot(history["data"]["Iter"], history["data"]["U norm"], "-o", label=caseName, clip_on=False)

ax.legend(labelcolor="linecolor")
niceplots.adjust_spines(ax)
plt.show()

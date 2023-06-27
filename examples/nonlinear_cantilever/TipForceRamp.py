"""
==============================================================================
Nonlinear cantilever beam analysis
==============================================================================
@File    :   analysis.py
@Date    :   2023/01/24
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import pickle

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
from pprint import pprint

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import pyTACS, constitutive, elements


# ==============================================================================
# Constants
# ==============================================================================
COMM = MPI.COMM_WORLD
PWD = os.path.dirname(__file__)
BDF_FILE = os.path.join(PWD, "Beam.bdf")
E = 1.2e6  # Young's modulus
NU = 0.0  # Poisson's ratio
RHO = 1.0  # density
YIELD_STRESS = 1.0  # yield stress
THICKNESS = 0.1  # Shell thickness
FORCE_MULTIPLIER = 4.0  # Multiplier applied to the baseline force of EI/L^2
MOMENT_MULTIPLIER = 0.1  # Multiplier applied to the baseline moment of 2pi * EI/L (which results in a full rotation)
STRAIN_TYPE = "linear"
ROTATION_TYPE = "linear"

elementType = None
if STRAIN_TYPE == "linear":
    if ROTATION_TYPE == "linear":
        elementType = elements.Quad4Shell
    elif ROTATION_TYPE == "quadratic":
        elementType = elements.Quad4ShellModRot
    elif ROTATION_TYPE == "quaternion":
        elementType = elements.Quad4ShellQuaternion
elif STRAIN_TYPE == "nonlinear":
    if ROTATION_TYPE == "linear":
        elementType = elements.Quad4NonlinearShell
    elif ROTATION_TYPE == "quadratic":
        elementType = elements.Quad4NonlinearShellModRot
    elif ROTATION_TYPE == "quaternion":
        elementType = elements.Quad4NonlinearShellQuaternion

if elementType is None:
    raise RuntimeError("Invalid element type, check STRAIN_TYPE and ROTATION_TYPE.")

# ==============================================================================
# Create pyTACS Assembler and problems
# ==============================================================================
structOptions = {
    "printtiming": True,
    "isNonlinear": True,
}
FEAAssembler = pyTACS(BDF_FILE, options=structOptions, comm=COMM)


def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    matProps = constitutive.MaterialProperties(rho=RHO, E=E, nu=NU, YS=YIELD_STRESS)
    con = constitutive.IsoShellConstitutive(
        matProps, t=THICKNESS, tNum=dvNum, tlb=1e-2 * THICKNESS, tub=1e2 * THICKNESS
    )
    transform = None
    element = elementType(transform, con)
    tScale = [10.0]
    return element, tScale


FEAAssembler.initialize(elemCallBack)

probOptions = {
    "printTiming": True,
    "skipFirstNLineSearch": 1,
    "continuationCoarseRelTol": 1e-3,
    "newtonSolverUseEW": True,
    "newtonSolverMaxLinIters": 10,
    "continuationInitialStep": 1.0,
    "continuationUsePredictor": True,
    "continuationNumPredictorStates": 7,
    "writeNLIterSolutions": False,
}
forceProblem = FEAAssembler.createStaticProblem("TipForce", options=probOptions)


# ==============================================================================
# Determine beam dimensions and other properties
# ==============================================================================
bdfInfo = FEAAssembler.getBDFInfo()
# cross-reference bdf object to use some of pynastrans advanced features
bdfInfo.cross_reference()
nodeCoords = bdfInfo.get_xyz_in_coord()
beamLength = np.max(nodeCoords[:, 0]) - np.min(nodeCoords[:, 0])
beamWidth = np.max(nodeCoords[:, 1]) - np.min(nodeCoords[:, 1])
I = beamWidth * THICKNESS**3 / 12.0

# ==============================================================================
# Add tip loads for each case
# ==============================================================================
tipForce = FORCE_MULTIPLIER * E * I / beamLength**2

# In order to work for different mesh sizes, we need to find the tip node IDs
# ourselves, we do this by finding the indices of the nodes whose x coordinate
# is within a tolerance of the max X coordinate in the mesh
tipNodeInds = np.nonzero(np.abs(np.max(nodeCoords[:, 0]) - nodeCoords[:, 0]) <= 1e-6)[0]
nastranNodeNums = list(bdfInfo.node_ids)
tipNodeIDs = [nastranNodeNums[ii] for ii in tipNodeInds]
numTipNodes = len(tipNodeIDs)

forceProblem.addLoadToNodes(
    tipNodeIDs, [0, 0, tipForce / numTipNodes, 0, 0, 0], nastranOrdering=True
)

# ==============================================================================
# Solve all problems and evaluate functions
# ==============================================================================
forceFactor = np.arange(0.05, 1.01, 0.05)
ForceVec = np.copy(forceProblem.F_array)

results = {"zDisp": [0.0], "xDisp": [0.0], "yRot": [0.0], "tipForce": [0.0]}

for scale in forceFactor:
    Fext = (scale-1.0) *ForceVec
    forceProblem.solve(Fext=Fext)

    fileName = f"{STRAIN_TYPE}_{ROTATION_TYPE}"
    forceProblem.writeSolution(outputDir=PWD, baseName=fileName)
    disps = forceProblem.u_array
    xDisps = disps[0::6]
    zDisps = disps[2::6]
    yRot = disps[4::6]
    results["tipForce"].append(scale)
    results["xDisp"].append(xDisps[-1])
    results["zDisp"].append(zDisps[-1])
    results["yRot"].append(yRot[-1])

for key in results:
    results[key] = np.array(results[key])

with open(os.path.join(PWD, f"TACS-Disps-{fileName}.pkl"), "wb") as f:
    pickle.dump(results, f)

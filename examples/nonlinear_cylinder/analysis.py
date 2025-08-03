"""
==============================================================================
Nonlinear cantilever beam analysis
==============================================================================
@File    :   analysis.py
@Date    :   2023/01/24
@Author  :   Alasdair Christison Gray
@Description : This code runs an analysis of a cantilever beam modeled with
shell elements subject to a vertical tip force. The problem is taken from
section 3.1 of "Popular benchmark problems for geometric nonlinear analysis of
shells" by Sze et al (https://doi.org/10.1016/j.finel.2003.11.001).
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import pickle
import argparse

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import pyTACS, constitutive, elements

# ==============================================================================
# Parse command line arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strainType", type=str, default="nonlinear", choices=["linear", "nonlinear"]
)
parser.add_argument(
    "--rotationType",
    type=str,
    default="quadratic",
    choices=["linear", "quadratic", "quaternion"],
)
parser.add_argument(
    "--incType", type=str, default="arcLength", choices=["arcLength", "load"]
)
args = parser.parse_args()

# ==============================================================================
# Constants
# ==============================================================================
COMM = MPI.COMM_WORLD
PWD = os.path.dirname(__file__)
BDF_FILE = os.path.join(PWD, "cylinder.bdf")
E = 3.10275e9  # Young's modulus
NU = 0.3  # Poisson's ratio
THICKNESS = 6.35e-3  # Shell thickness

RHO = 1.0  # density
YIELD_STRESS = 1.0  # yield stress
MAX_FORCE = 3000.0  # Multiplier applied to the baseline force of EI/L^2
STRAIN_TYPE = args.strainType
ROTATION_TYPE = args.rotationType

# Overall dimensions
radius = 2.540
length = radius / 10
angle = 0.1  # radians

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
}
FEAAssembler = pyTACS(BDF_FILE, options=structOptions, comm=COMM)


def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    matProps = constitutive.MaterialProperties(rho=RHO, E=E, nu=NU, ys=YIELD_STRESS)
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
    "printLevel": 1,
    "nonlinearIncType": "ArcLength" if args.incType == "arcLength" else "Load",
}
forceProblem = FEAAssembler.createStaticProblem("PointForce", options=probOptions)

if forceProblem.isNonlinear:
    newtonOptions = {"useEW": True, "MaxLinIters": 10}
    continuationOptions = {
        "CoarseRelTol": 1e-3,
        "InitialStep": 0.05,
        "UsePredictor": True,
        "NumPredictorStates": 7,
    }
    arcLengthOptions = {"eta": 0.10}
    arcLengthOptions.update(continuationOptions)
    if args.incType == "arcLength":
        forceProblem.nonlinearSolver.setOptions(arcLengthOptions)
    else:
        forceProblem.nonlinearSolver.setOptions(continuationOptions)
        forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)

# ==============================================================================
# Add point load
# ==============================================================================
# In order to work for different mesh sizes, we need to find the tip node IDs
# ourselves, we do this by finding the indices of the nodes whose x coordinate
# is within a tolerance of the max X coordinate in the mesh
bdfInfo = FEAAssembler.getBDFInfo()
# cross-reference bdf object to use some of pynastrans advanced features
bdfInfo.cross_reference()
nodeCoords = bdfInfo.get_xyz_in_coord()

# Find the node whose coordinate is closest to (0, 0, radius)
dist = np.linalg.norm(nodeCoords - np.array([0, 0, radius]), axis=1)
forceNodeInd = np.argmin(dist)
nastranNodeNums = list(bdfInfo.node_ids)
tipNodeID = nastranNodeNums[forceNodeInd]

forceProblem.addLoadToNodes(
    tipNodeID, [0, 0, -MAX_FORCE / 4, 0, 0, 0], nastranOrdering=True
)

# ==============================================================================
# Some pre-computation to help us extract the centre displacement
# ==============================================================================

tipNodeIDLocal = FEAAssembler.meshLoader.getLocalNodeIDsFromGlobal(
    [tipNodeID], nastranOrdering=True
)[0]
hasCentreDisp = tipNodeIDLocal != -1
dispNodeRank = np.argmax(hasCentreDisp)


def getCentreDisp():
    centreDisp = None
    if tipNodeIDLocal != -1:
        centreDisp = forceProblem.u_array[tipNodeIDLocal * 6 + 2]
    centreDisp = forceProblem.comm.bcast(centreDisp, root=dispNodeRank)
    return centreDisp


results = {"zDisp": [0.0], "loadScale": [0.0]}


fileName = f"{STRAIN_TYPE}_{ROTATION_TYPE}_{args.incType}-Incrementation"

if args.incType == "load":
    # ==============================================================================
    # Run analysis with load scales in 5% increments from 5% to 100%
    # ==============================================================================
    stepSize = 0.025
    forceFactor = np.arange(0.025, 1.01, 0.025)
    ForceVec = np.copy(forceProblem.F_array)

    for scale in forceFactor:
        Fext = (scale - 1.0) * ForceVec
        forceProblem.solve(Fext=Fext)

        forceProblem.writeSolution(outputDir=PWD, baseName=fileName)
        results["zDisp"].append(getCentreDisp())
        results["loadScale"].append(scale)

else:
    # Set an increment callback to write the solution at each increment
    def incrementCallback(solver, u, resVec, monitorVars):
        # Write the solution to a file
        forceProblem.writeSolution(
            baseName=f"{STRAIN_TYPE}_{ROTATION_TYPE}_{args.incType}-Incrementation-{solver.iterationCount}"
        )
        results["zDisp"].append(getCentreDisp())
        results["loadScale"].append(forceProblem.loadScale)
        # return abs(results["zDisp"][-1]) > 0.02

    forceProblem.nonlinearSolver.setIncrementCallback(incrementCallback)
    forceProblem.solve()

# Solve has finished, write the equilibrium path to a file
if forceProblem.comm.rank == 0:
    print(f"{results=}")

    for key in results:
        results[key] = np.array(results[key])

    with open(os.path.join(PWD, f"TACS-Disps-{fileName}.pkl"), "wb") as f:
        pickle.dump(results, f)

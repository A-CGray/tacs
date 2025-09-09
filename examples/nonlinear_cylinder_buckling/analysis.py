"""
==============================================================================
Cylinder buckling analysis example
==============================================================================
@File    :   analysis.py
@Date    :   2025/08/03
@Author  :   Alasdair Christison Gray
@Description : Analysis of cylinder buckling using arc-length solver
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
BDF_FILE = os.path.join(PWD, "mech-cylinder.bdf")
RHO = 2718.0  # density
E = 72.0e9  # Young's modulus
NU = 0.33  # Poisson's ratio
RADIUS = 0.2
THICKNESS_RATIO = 100 # radius to thickness ratio
THICKNESS = RADIUS / THICKNESS_RATIO  # thickness of the shell
LENGTH = 0.4
YIELD_STRESS = 1.0  # yield stress

STRAIN_TYPE = args.strainType
ROTATION_TYPE = args.rotationType

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

# Setup a buckling problem
bucklingProblem  = FEAAssembler.createBucklingProblem("CompressionBuckling", sigma=50.0, numEigs=20)
bucklingProblem.setOption("printLevel", 2)
bucklingProblem.solve()
bucklingProblem.writeSolution()


# Run a nonlinear static analysis with a small imperfection in the shape of the first 5 buckling modes
imperfectionSize = THICKNESS*0.05
AssemblerCoords = FEAAssembler.Xpts0.getArray()
for mode in range(5):
    eigVal, eigVec = bucklingProblem.getVariables(mode)
    eigVecDisps = eigVec.reshape(-1, 6)[:,:3]  # Reshape to displacements
    # Get the max displacement in the mode shape
    maxDisp = bucklingProblem.comm.allreduce(np.max(np.abs(np.linalg.norm(eigVecDisps[:,1:],axis=1))), op=MPI.MAX)
    # Scale the mode shape by a small factor relative to the max displacement
    eigVecDisps *= imperfectionSize / maxDisp
    AssemblerCoords += eigVecDisps.flatten()


probOptions = {
    "printTiming": True,
    "printLevel": 1,
    "nonlinearIncType": "ArcLength" if args.incType == "arcLength" else "Load",
}
staticProblem = FEAAssembler.createStaticProblem("Compression", options=probOptions)
staticProblem.writeSolution(baseName="Initial-Coords")
# exit()

if staticProblem.isNonlinear:
    newtonOptions = {"useEW": True, "MaxLinIters": 10, "SkipFirstNLineSearch":1}
    continuationOptions = {
        "CoarseRelTol": 1e-3,
        "InitialStep": 0.05,
        "UsePredictor": True,
        "NumPredictorStates": 7,
    }
    arcLengthOptions = {"eta": 0.0, "MaxLambda":2000.0, "InitialStep": 2.0, "MaxIter": 200, "RelTol": 1e-6, "AbsTol": 1e-6}
    arcLengthOptions.update(continuationOptions)
    if args.incType == "arcLength":
        staticProblem.nonlinearSolver.setOptions(arcLengthOptions)

        def incrementCallback(solver, u, resVec, monitorVars):
            # Write the solution to a file
            staticProblem.writeSolution(
                baseName=f"{STRAIN_TYPE}_{ROTATION_TYPE}_{args.incType}-Incrementation-{solver.iterationCount}"
            )
        staticProblem.nonlinearSolver.setIncrementCallback(incrementCallback)
    else:
        staticProblem.nonlinearSolver.setOptions(continuationOptions)
        staticProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)

staticProblem.solve()
staticProblem.writeSolution()

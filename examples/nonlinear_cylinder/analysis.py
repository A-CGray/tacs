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
from tacs import pyTACS, constitutive, elements, TACS

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
parser.add_argument("--incType", type=str, default="arcLength", choices=["arcLength", "load"])
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
}
newtonOptions = {"useEW": True, "MaxLinIters": 10}
continuationOptions = {
    "CoarseRelTol": 1e-3,
    "InitialStep": 0.05,
    "UsePredictor": True,
    "NumPredictorStates": 7,
}
forceProblem = FEAAssembler.createStaticProblem("PointForce", options=probOptions)
try:
    forceProblem.nonlinearSolver.innerSolver.setOptions(newtonOptions)
    forceProblem.nonlinearSolver.setOptions(continuationOptions)
except AttributeError:
    pass

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

if args.incType == "load":
    # ==============================================================================
    # Run analysis with load scales in 5% increments from 5% to 100%
    # ==============================================================================
    forceFactor = np.arange(0.0, 1.01, 0.05)
    ForceVec = np.copy(forceProblem.F_array)

    results = {"zDisp": [0.0], "xDisp": [0.0], "yRot": [0.0], "tipForce": [0.0]}

    for scale in forceFactor:
        Fext = (scale - 1.0) * ForceVec
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

else:
    constraintType = "nonlinear"
    maxInc = 1000
    maxIter = 20
    forceProblem.zeroVariables()
    u = forceProblem.u
    du = FEAAssembler.createVec(asBVec=True)
    incStartDisp = FEAAssembler.createVec(asBVec=True)
    incStartDisp.copyValues(u)
    loadFactor = 0.0
    s = 0.0
    tol = 1e-9

    dsInit = 0.05
    eta = 0.0
    minStep = 0.001
    maxStep = 1.0
    nIterDes = 4
    maxLoadFactor = 1.0

    loadFactorHist = [0.0]

    # Compute external force vector
    Fex = FEAAssembler.createVec(asBVec=True)
    Fin = FEAAssembler.createVec(asBVec=True)
    forceProblem.getForces(Fex, Fin)
    Fex.scale(-1.0)
    FexNorm = Fex.norm()

    # Create continuation path matrix
    dConstraintdu = FEAAssembler.createVec(asBVec=True)
    dConstraintdLambda = 0.
    pathMat = TACS.ContinuationPathMat(forceProblem.K, Fex, dConstraintdu, dConstraintdLambda)
    pathSolver = TACS.KSM(
                    pathMat,
                    forceProblem.PC,
                    forceProblem.getOption("subSpaceSize"),
                    forceProblem.getOption("nRestarts"),
                    forceProblem.getOption("flexible"),
                )
    ds = 0.0

    # Create other required vectors
    prevIncStep = FEAAssembler.createVec(asBVec=True)
    tangentStep = FEAAssembler.createVec(asBVec=True)
    incStartDisp = FEAAssembler.createVec(asBVec=True)

    for increment in range(maxInc):
        prevIncStep.copyValues(du)
        incStartDisp.copyValues(u)
        incStartLoadFactor = loadFactor

        # Compute initial guess for the next increment
        forceProblem.updateJacobian()
        forceProblem.updatePreconditioner()
        forceProblem.linearSolver.solve(Fex, tangentStep)
        tangentStep.scale(-1.0)
        tangentNorm2 = tangentStep.norm()**2

        # If this is the first increment, compute the initial arc length step size, interpret the user's input as the desired change in the load factor in the first increment
        if increment == 0:
            ds = np.sqrt(dsInit**2 * (eta + tangentNorm2))
            dsMin = np.sqrt(minStep**2 * (eta + tangentNorm2))
            dsMax = np.sqrt(maxStep**2 * (eta + tangentNorm2))
        dLoadFactor = ds / np.sqrt(eta + tangentNorm2)

        # Choose between the positive and negative roots of the constraint equation
        if increment > 0:
            if prevIncStep.dot(tangentStep) < 0:
                dLoadFactor *= -1

        # Take the tangent step
        loadFactor += dLoadFactor
        tangentStep.scale(dLoadFactor)
        u.axpy(1.0, tangentStep)

        du.copyValues(u)
        du.axpy(-1.0, incStartDisp)
        dy = loadFactor - incStartLoadFactor

        # Now do a Newton solve to find the equilibrium state
        innerSolverConverged = False
        for innerIter in range(maxIter):
            forceProblem.setVariables(u)
            forceProblem.setLoadScale(loadFactor)

            # Compute the residual
            forceProblem.getResidual(forceProblem.res)
            resNorm = forceProblem.res.norm() / abs(loadFactor * FexNorm)

            # Compute the arc-length constraint g = sqrt(du^T du + eta dy^2) - ds
            radius = np.sqrt(du.norm()**2 + eta * dy**2)
            constraint = radius - ds
            print(
                f"Increment: {increment:02d}, Iteration: {innerIter:02d}, LoadFactor: {loadFactor: .6e}, Residual: {resNorm: .6e}, Constraint: {constraint: .6e}, uNorm: {u.norm(): .6e}"
            )
            if constraintType.lower() == "nonlinear":
                innerSolverConverged = resNorm < tol and np.abs(constraint) < tol
            elif constraintType.lower() == "linear":
                innerSolverConverged = resNorm < tol
            if innerSolverConverged:
                break

            # Build then solve the N+1 system matrix
            # [ KT   | -Fex ] [ du ] = [ -res ]
            # [ dgdu | dgdy ] [ dy ] = [ -constraint ]
            forceProblem.updateJacobian()
            forceProblem.updatePreconditioner()
            forceProblem.res.scale(-1.0)

            if constraintType.lower() == "nonlinear":
                dConstraintdu.copyValues(du)  # ddu(sqrt(du^T du + eta dy^2) - ds) = du / sqrt(du^T du + eta dy^2)
                dConstraintdu.scale(1 / radius)
                dConstraintdLambda = eta * dy / radius
                pathMat.setConstraint(dConstraintdLambda)
                if constraint != 0:
                    tBarNorm2 = dConstraintdu.norm()**2 + dConstraintdLambda**2
                    a = -constraint / tBarNorm2
                    forceProblem.K.mult(dConstraintdu, forceProblem.update)
                    forceProblem.update.axpy(dConstraintdLambda, Fex)
                    forceProblem.res.axpy(-a, forceProblem.update)
                else:
                    a = 0.
                pathSolver.solve(forceProblem.res, forceProblem.update)
                loadScaleUpdate = pathMat.applyQ(forceProblem.update)
                forceProblem.update.axpy(a, dConstraintdu)
                loadScaleUpdate += a * dConstraintdLambda
            elif constraintType.lower() == "linear":
                if innerIter == 0:
                    dConstraintdu.copyValues(tangentStep)
                    pathMat.setConstraint(dLoadFactor)
                pathSolver.solve(forceProblem.res, forceProblem.update)
                loadScaleUpdate = pathMat.applyQ(forceProblem.update)

            # Limit any step that is bigger than the arc length constraint radius
            alpha = 1.0
            stepSize = alpha * np.sqrt(forceProblem.update.norm()**2 + eta * loadScaleUpdate ** 2)
            if stepSize > ds:
                print("Limiting step size")
                alpha *= ds / stepSize
            u.axpy(alpha, forceProblem.update)
            loadFactor += alpha * loadScaleUpdate

            # Update the displacement and load factor change for the current increment
            du.copyValues(u)
            du.axpy(-1.0, incStartDisp)
            dy = loadFactor - incStartLoadFactor

        # End of increment, check if we should accept the step, we shouldn't accept if:
        # 1. The inner solver didn't converge
        # 2. The computed step is in the opposite direction of the initial tangent step for this increment
        rejectIncrement = not innerSolverConverged
        if not rejectIncrement:
            stepCosine = du.dot(tangentStep) / (
                du.norm() * tangentStep.norm()
            )
            rejectIncrement = stepCosine <= 0  # np.cos(np.pi / 16)
        if rejectIncrement:
            u.copyValues(incStartDisp)
            loadFactor = incStartLoadFactor
            ds *= 0.5
            du.copyValues(prevIncStep)
            print("Step rejected")
        else:
            forceProblem.writeSolution(baseName=f"{forceProblem.name}_{increment:04d}")
            loadFactorHist.append(loadFactor)
            ds *= np.clip(np.sqrt(nIterDes / (innerIter)), 0.25, 4.0)
            ds = np.clip(ds, dsMin, dsMax)
            print("Step accepted")
            if abs(loadFactor) > maxLoadFactor:
                break

"""
==============================================================================
TACS Nonlinear Arc-Length Solver
==============================================================================
This solver uses the arc-length method to solve nonlinear problems.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from typing import Optional, Callable, Any, Dict, Union
import copy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import mpi4py

# ==============================================================================
# Extension modules
# ==============================================================================
import tacs.TACS
from tacs.solvers import BaseSolver
from tacs.solvers.utils import lagrangeInterp


class ArcLengthSolver(BaseSolver):
    defaultOptions = {
        "UseLinearConstraint": [
            bool,
            False,
            "By default, the Arc-Length solver will use the true nonlinear arc length constraint. If set to true, the solver uses a linearised version of the constraint, where the increment solution is constrained to lie on a line orthogonal to the initial tangent step, this is also known as the Riks method.",
        ],
        "MaxLambda": [
            float,
            1.0,
            "Final continuation parameter value to aim for.",
        ],
        "eta": [
            float,
            0.0,
            "Arc-length constraint parameter. Controls the weight of the load factor change in the arc-length constraint. The eta value given by the user is interpreted as the desired contribution of the load factor change to the arc-length of the first tangent step, relative to the displacement change. In other words, the weight, w, is chosen such that w*dLambda = eta * ||du||^2, where ||du|| is the norm of the displacement change in the first tangent step. eta=0 corresponds to the cylindrical arc-length method, eta->inf corresponds to pure load incrementation.",
        ],
        "AbsTol": [
            float,
            1e-8,
            "Convergence criteria for the nonlinear residual norm.",
        ],
        "RelTol": [
            float,
            1e-8,
            "Relative convergence criteria for the nonlinear residual norm, norm is measured relative to that of the external load vector.",
        ],
        "DivergenceTol": [
            float,
            1e10,
            "Residual norm at which the nonlinear solver is jugded to have diverged",
        ],
        "CoarseAbsTol": [
            float,
            1e-8,
            "Residual norm criteria for intermediate increments, making this larger may speed up the nonlinear solver by allowing it to only partially converge intermediate steps.",
        ],
        "CoarseRelTol": [
            float,
            1e-8,
            "Relative residual norm criteria for intermediate increments.",
        ],
        "TargetIter": [
            int,
            8,
            "Target number of Newton iterations for each increment.",
        ],
        "MaxIter": [int, 30, "Maximum number of increments."],
        "InitialStep": [float, 0.2, "Target initial load factor for first increment."],
        "MinStep": [float, 1e-4, "Minimum arc-length step size."],
        "MaxStep": [float, np.inf, "Maximum arc-length step size."],
        "MinStepFactor": [
            float,
            0.5,
            "The minimum factor by which the continuation step size can decrease in a single step.",
        ],
        "MaxStepFactor": [
            float,
            2.0,
            "The maximum factor by which the continuation step size can increase in a single step.",
        ],
        "RetractionFactor": [
            float,
            0.5,
            "The factor by which the continuation step size is reduced when the Newton solver fails to converge.",
        ],
        # Predictor step options
        "UsePredictor": [
            bool,
            False,
            "Flag for using predictor step in continuation.",
        ],
        "NumPredictorStates": [
            int,
            3,
            "Number of previous equilibrium states to use in computing the predictor step.",
        ],
    }

    def __init__(
        self,
        tangentSolver: tacs.TACS.KSM,
        pathSolver: tacs.TACS.KSM,
        jacMat: tacs.TACS.Mat,
        pc: tacs.TACS.Pc,
        setLambdaFunc: Callable,
        getLambdaFunc: Callable,
        createVecFunc: Callable,
        setStateFunc: Callable,
        jacUpdateFunc: Callable,
        pcUpdateFunc: Callable,
        resFunc: Callable,
        options: Optional[dict] = None,
        comm: Optional[mpi4py.MPI.Comm] = None,
    ) -> None:
        """Create a continuation solver instance
        Parameters
        ----------
        tangentSolver : tacs.TACS.KSM
            The linear solver to be used to solve for the tangent stiffness matrix
        pathSolver : tacs.TACS.KSM
            The linear solver to be used to solve for the augmented arc-length matrix
        jacMat : tacs.TACS.Mat
            The Jacobian matrix
        pc : tacs.TACS.Pc
            The preconditioner
        setLambdaFunc : function
            Function to set the continuation parameter, with signature `setLambdaFunc(lambda:float) -> None`
        getLambdaFunc : function
            Function to get the current continuation parameter, with signature `getLambdaFunc() -> float`
        createVecFunc : function
            Function to create a new TACS vector, with signature `createVecFunc() -> tacs.TACS.Vec`
        setStateFunc : function
            Function to set the current state of the system, with signature `setStateFunc(state: tacs.TACS.Vec) -> None`
        jacUpdateFunc : function
            Function to update the residual Jacobian at the current state, with signature `jacUpdateFunc() -> None`
        pcUpdateFunc : function
            Function to update the residual Jacobian preconditioner at the current state, with signature `pcUpdateFunc() -> None`
        resFunc : function
            Function to compute the residual at the current state, with signature `resFunc(res: tacs.TACS.Vec) -> None`
        options : dict, optional
            Dictionary holding solver-specific option parameters (case-insensitive)., by default None
        comm : mpi4py.MPI.Intracomm, optional
            The comm object on which to create the pyTACS object., by default mpi4py.MPI.COMM_WORLD
        """
        self.tangentSolver = tangentSolver
        self.pathSolver = pathSolver
        self.jacMat = jacMat
        self.jacUpdateFunc = jacUpdateFunc
        self.pcUpdateFunc = pcUpdateFunc
        self.setLambdaFunc = setLambdaFunc
        self.getLambdaFunc = getLambdaFunc

        self.equilibriumPathStates: list[tacs.TACS.Vec] = []
        self.equilibriumPathLoadScales: list[Union[float, None]] = []
        self.equilibriumPathLengths: list[Union[float, None]] = []

        self.incrementCallback: Optional[Callable] = None

        BaseSolver.__init__(
            self,
            resFunc=resFunc,
            createVecFunc=createVecFunc,
            setStateFunc=setStateFunc,
            options=options,
            comm=comm,
        )

        # Create additional vectors
        self.fInt = self.createVecFunc()
        self.fExt = self.createVecFunc()
        self.du_e = self.createVecFunc()
        self.du_i = self.createVecFunc()
        self.incStartState = self.createVecFunc()
        self.du = self.createVecFunc()
        self.prevIncStep = self.createVecFunc()
        self.tangentStep = self.createVecFunc()
        self.incStartDisp = self.createVecFunc()
        self.Fex = self.createVecFunc()
        self.Fin = self.createVecFunc()
        self.dgdu = self.createVecFunc()
        self.update = self.createVecFunc()

        # Create continuation path matrix
        dgdLambda = 0.0
        self.pathMat = tacs.TACS.ContinuationPathMat(
            jacMat, self.Fex, self.dgdu, dgdLambda
        )
        self.pathSolver.setOperators(self.pathMat, pc)

    def updateOperators(self, mat: tacs.TACS.Mat, pc: tacs.TACS.Pc) -> None:
        """
        Update the matrix and preconditioner operators used by the solver.

        Parameters
        ----------
        mat : tacs.TACS.Mat
            The matrix operator
        pc : tacs.TACS.Pc
            The preconditioner
        """
        self.jacMat = mat
        # Update the tangent solver with the new operators
        self.tangentSolver.setOperators(mat, pc)

        # The Jacobian within pathMat is immutable, so we must create a new instance
        # using the new matrix. We can reuse the other vectors.
        dgdLambda = (
            self.pathMat.getConstraint()
        )  # Preserve the existing constraint value
        self.pathMat = tacs.TACS.ContinuationPathMat(
            mat, self.Fex, self.dgdu, dgdLambda
        )

        # Update the path solver to use the new path matrix and preconditioner
        self.pathSolver.setOperators(self.pathMat, pc)

    def setIncrementCallback(
        self, incrementCallback: Optional[Callable] = None
    ) -> Optional[bool]:
        """
        Set the user-defined callback function to be called at the end of each successful increment.

        Parameters
        ----------
        incrementCallback : callable, optional
            The user-defined callback function. The callback function should have the following signature:
            `callback(solver: BaseSolver, u: tacs.TACS.Vec, res: tacs.TACS.Vec, monitorVars: dict) -> Optional[bool]`
            If the function returns True, the solver will terminate the solution, you can use this to implement your own
            termination criteria.
        """
        self.incrementCallback = incrementCallback

    def getHistoryVariables(self) -> Dict[str, Dict]:
        """Get the variables to be stored in the solver history

        This method allows for implementation of any logic that dictates any changes in the stored variables depending on the current options.

        Returns
        -------
        Dict[str, Dict]
            Dictionary of solver variables, keys are the variable names, value is another dictionary with keys "type" and "print", where "type" is the data type of the variable and "print" is a boolean indicating whether or not to print the variable to the screen
        """
        numType = float if self.dtype == np.float64 else complex

        variables = {}
        variables["Increment"] = {"type": int, "print": True}
        variables["Lambda"] = {"type": float, "print": True}
        variables["SubIter"] = {"type": int, "print": True}
        # Residual norm (absolute and relative)
        variables["Res norm"] = {"type": numType, "print": True}
        variables["Rel res norm"] = {"type": numType, "print": True}
        # Arc-length constraint
        variables["Constraint"] = {"type": numType, "print": True}
        # state norm
        variables["U norm"] = {"type": numType, "print": True}
        # Displacement step norm
        variables["du norm"] = {"type": numType, "print": True}
        # Load factor step norm
        variables["dLambda"] = {"type": numType, "print": True}
        variables["Flags"] = {"type": str, "print": True}

        return variables

    def _setupPredictorVectors(self) -> None:
        """Setup the structures containing the data for computing the predictor steps"""
        self.equilibriumPathStates = []
        self.equilibriumPathLoadScales = []
        self.equilibriumPathLengths = []
        if self.getOption("UsePredictor"):
            for _ in range(self.getOption("NumPredictorStates")):
                self.equilibriumPathStates.append(self.createVecFunc())
                self.equilibriumPathLoadScales.append(None)
                self.equilibriumPathLengths.append(None)

    def setOption(self, name: str, value: Any) -> None:
        BaseSolver.setOption(self, name, value)

        # Update the predictor computation data structures if the relevant options are changed
        if name.lower() in [
            "usepredictor",
            "numpredictorstates",
        ]:
            self._setupPredictorVectors()

    def setConvergenceTolerances(
        self, absTol: Optional[float] = None, relTol: Optional[float] = None
    ) -> None:
        """Set the convergence tolerance of the solver

        Parameters
        ----------
        absTol : float, optional
            Absolute tolerance, not changed if no value is provided
        relTol : float, optional
            Relative tolerance, not changed if no value is provided
        """
        if absTol is not None:
            self.setOption("AbsTol", absTol)
        if relTol is not None:
            self.setOption("RelTol", relTol)

        return

    def initializeSolve(self, u0: Optional[tacs.TACS.Vec] = None) -> None:
        """Perform any initialization required before the solve

        For now we do not try to restart the arc-length solver from a previous solution, this could be added in the future.
        """
        BaseSolver.initializeSolve(self)
        if u0 is not None:
            Warning(
                "ArcLengthSolver does not support restarting from a previous solution, ignoring the provided initial state."
            )

        self.stateVec.zeroEntries()
        self.setStateFunc(self.stateVec)
        self.setLambdaFunc(0.0)
        if self.getOption("UsePredictor"):
            for ii in range(self.getOption("NumPredictorStates")):
                self.equilibriumPathLoadScales[ii] = None
                self.equilibriumPathLengths[ii] = None
                self.equilibriumPathStates[ii].zeroEntries()

    def solve(
        self, u0: Optional[tacs.TACS.Vec] = None, result: Optional[tacs.TACS.Vec] = None
    ) -> None:
        MAX_LAMBDA = self.getOption("MaxLambda")
        TARGET_ITERS = self.getOption("TargetIter")
        INIT_STEP = self.getOption("InitialStep")
        MIN_STEP = self.getOption("MinStep")
        MAX_STEP = self.getOption("MaxStep")
        MAX_INCREMENTS = self.getOption("MaxIter")
        MIN_STEP_FACTOR = self.getOption("MinStepFactor")
        MAX_STEP_FACTOR = self.getOption("MaxStepFactor")
        STEP_RETRACT_FACTOR = self.getOption("RetractionFactor")
        MAX_RES = self.getOption("DivergenceTol")
        USE_LIN_CONSTRAINT = self.getOption("UseLinearConstraint")

        # ABS_TOL = self.getOption("AbsTol")
        # REL_TOL = self.getOption("RelTol")
        # COARSE_ABS_TOL = self.getOption("CoarseAbsTol")
        # COARSE_REL_TOL = self.getOption("CoarseRelTol")

        # USE_PREDICTOR = self.getOption("UsePredictor")

        self.initializeSolve(u0)
        maxIter = 20

        u = self.stateVec
        self.incStartDisp.copyValues(u)
        loadFactor = 0.0
        s = 0.0
        tol = 1e-9

        # Compute external force vector
        self.computeForceVectors()
        self.Fex.copyValues(self.fExt)
        FexNorm = self.Fex.norm()
        if FexNorm > 0:
            self.setRefNorm(FexNorm)

        ds = 0.0

        flags = ""
        finalIncrement = False
        for increment in range(MAX_INCREMENTS):
            self._iterationCount = increment
            self.prevIncStep.copyValues(self.du)
            self.incStartDisp.copyValues(u)
            incStartLoadFactor = loadFactor

            # Compute initial guess for the next increment
            self.jacUpdateFunc()
            self.pcUpdateFunc()
            self.tangentSolver.solve(self.Fex, self.tangentStep)
            self.tangentStep.scale(-1.0)
            tangentNorm2 = self.tangentStep.norm() ** 2

            # Interpret the user's eta value as how much the load factor change should be weighted in the first step arc-length relative to the displacement change
            ETA = self.getOption("eta") * tangentNorm2

            # If this is the first increment, compute the initial arc length step size, interpret the user's input as the desired change in the load factor in the first increment
            if increment == 0:
                ds = np.sqrt(INIT_STEP**2 * (ETA + tangentNorm2))
                dsMin = np.sqrt(MIN_STEP**2 * (ETA + tangentNorm2))
                dsMax = np.sqrt(MAX_STEP**2 * (ETA + tangentNorm2))
            dLoadFactor = ds / np.sqrt(ETA + tangentNorm2)

            # Limit the load factor step size if we're predicted to go way past the maximum load factor
            if loadFactor + dLoadFactor > MAX_LAMBDA * 1.05:
                shrinkFactor = (MAX_LAMBDA * 1.05 - loadFactor) / dLoadFactor
                dLoadFactor *= shrinkFactor
                ds *= shrinkFactor

            # Choose between the positive and negative roots of the constraint equation
            if increment > 0:
                if self.prevIncStep.dot(self.tangentStep) < 0:
                    dLoadFactor *= -1

            # Take the tangent step
            loadFactor += dLoadFactor
            self.tangentStep.scale(dLoadFactor)
            u.axpy(1.0, self.tangentStep)

            self.du.copyValues(u)
            self.du.axpy(-1.0, self.incStartDisp)
            dy = loadFactor - incStartLoadFactor

            # Now do a Newton solve to find the equilibrium state
            innerSolverConverged = False
            innerSolverDiverged = False
            for innerIter in range(maxIter):
                self.setStateFunc(u)
                self.setLambdaFunc(loadFactor)

                # Compute the residual
                self.resFunc(self.resVec)
                resNorm = self.resVec.norm()
                relResNorm = resNorm / self.refNorm if self.refNorm > 0 else resNorm

                # Compute the arc-length constraint g = sqrt(du^T du + eta dy^2) - ds
                duNorm = self.du.norm()
                radius = np.sqrt(duNorm**2 + ETA * dy**2)
                constraint = radius - ds
                uNorm = u.norm()

                # Check convergence/divergence
                if USE_LIN_CONSTRAINT:
                    innerSolverConverged = relResNorm < tol
                else:
                    innerSolverConverged = relResNorm < tol and np.abs(constraint) < tol

                innerSolverDiverged = np.real(resNorm) >= MAX_RES or np.isnan(resNorm)

                if innerSolverConverged:
                    flags += "C"
                elif innerSolverDiverged:
                    flags += "D"

                monitorVars = {
                    "Increment": increment,
                    "SubIter": innerIter,
                    "Lambda": loadFactor,
                    "Res norm": resNorm,
                    "Rel res norm": relResNorm,
                    "Constraint": constraint,
                    "U norm": uNorm,
                    "du norm": duNorm,
                    "dLambda": dy,
                    "Flags": flags,
                }

                if self.rank == 0:
                    self.history.write(monitorVars)

                if self.iterationCallback is not None:
                    self.iterationCallback(
                        self, self.stateVec, self.resVec, monitorVars
                    )
                flags = ""

                if innerSolverConverged or innerSolverDiverged:
                    break

                self.jacUpdateFunc()
                self.pcUpdateFunc()
                self.resVec.scale(-1.0)

                if USE_LIN_CONSTRAINT:
                    if innerIter == 0:
                        self.dgdu.copyValues(self.tangentStep)
                        self.pathMat.setConstraint(dLoadFactor)
                    self.pathSolver.solve(self.resVec, self.update)
                    loadScaleUpdate = self.pathMat.applyQ(self.update)
                else:
                    self.dgdu.copyValues(
                        self.du
                    )  # ddu(sqrt(du^T du + eta dy^2) - ds) = du / sqrt(du^T du + eta dy^2)
                    self.dgdu.scale(1 / radius)
                    dgdLambda = ETA * dy / radius
                    self.pathMat.setConstraint(dgdLambda)
                    if constraint != 0:
                        tBarNorm2 = self.dgdu.norm() ** 2 + dgdLambda**2
                        a = -constraint / tBarNorm2
                        self.jacMat.mult(self.dgdu, self.update)
                        self.update.axpy(dgdLambda, self.Fex)
                        self.resVec.axpy(-a, self.update)
                    else:
                        a = 0.0
                    self.pathSolver.solve(self.resVec, self.update)
                    loadScaleUpdate = self.pathMat.applyQ(self.update)
                    self.update.axpy(a, self.dgdu)
                    loadScaleUpdate += a * dgdLambda

                # Limit any step that is bigger than the arc length constraint radius
                alpha = 1.0
                stepSize = alpha * np.sqrt(
                    self.update.norm() ** 2 + ETA * loadScaleUpdate**2
                )
                if stepSize > ds:
                    flags += "L"
                    alpha *= ds / stepSize
                u.axpy(alpha, self.update)
                loadFactor += alpha * loadScaleUpdate

                # Update the displacement and load factor change for the current increment
                self.du.copyValues(u)
                self.du.axpy(-1.0, self.incStartDisp)
                dy = loadFactor - incStartLoadFactor

            # End of increment, check if we should accept the step, we shouldn't accept if:
            # 1. The inner solver didn't converge
            # 2. The computed step is in the opposite direction of the initial tangent step for this increment
            rejectIncrement = not innerSolverConverged
            if not rejectIncrement:
                # Take the dot product of the converged step with the initial tangent step
                dot = self.du.dot(self.tangentStep) + dLoadFactor * dy
                stepCosine = dot / (
                    np.sqrt(self.du.norm() ** 2 + dy**2)
                    * np.sqrt(self.tangentStep.norm() ** 2 + dLoadFactor**2)
                )
                rejectIncrement = stepCosine <= 0  # np.cos(np.pi / 16)
            if rejectIncrement:
                u.copyValues(self.incStartDisp)
                loadFactor = incStartLoadFactor
                ds *= STEP_RETRACT_FACTOR
                self.du.copyValues(self.prevIncStep)
                flags += "R"
                if self.comm.rank == 0:
                    print("Step rejected")
            else:
                # Before we take this step, we should check if it will take us past the maximum load factor. If it does,
                # we should set the load factor to the maximum and then solve for the displacements at that load factor.
                # We can generate a good initial guess for the displacements by interpolating the previous equilibrium
                # states.
                if loadFactor < MAX_LAMBDA:
                    s += ds
                    ds *= np.clip(
                        np.sqrt(TARGET_ITERS / (innerIter)),
                        MIN_STEP_FACTOR,
                        MAX_STEP_FACTOR,
                    )
                    ds = np.clip(ds, dsMin, dsMax)
                    if self.incrementCallback is not None:
                        terminate = self.incrementCallback(self, u, self.resVec, monitorVars)
                        if terminate:
                            break
                elif loadFactor > MAX_LAMBDA:
                    self.incrementCallback(self, u, self.resVec, monitorVars)
                    break
                    # TODO: Implement fixed load factor solve that relies on this code.
                    fraction = (MAX_LAMBDA - incStartLoadFactor) / dy
                    u.copyValues(self.incStartDisp)
                    u.axpy(fraction, self.du)
                    loadFactor = MAX_LAMBDA
                    finalIncrement = True

    def solveContinuationSystem(
        self,
        pathMat: tacs.TACS.ContinuationPathMat,
        rhsVec: tacs.TACS.Vec,
        rhsScalar: Optional[float] = None,
    ) -> None:
        """Solve a special type linear system that occurs in the arc-length continuation method:

        [ KT   | -Fex ] [ delta u ] = [ rhsVec ]
        [ dgdu | dgdy ] [ delta y ] = [ rhsScalar ]

        This system comes from linearising the equilibrium equations and arc-length constraint around the current state:
        r(u, lambda) = F_int(u) + lambda * F_ext(u, lambda)
        g(du, dy) = sqrt(du^T du + eta * dy^2) - ds

        Where:
        - r is the equilibrium residual
        - g is the arc-length constraint
        - du is the displacement step being solved for in the current increment
        - delta u is the change in du being solved for in this linear solve (Newton iteration)


        _extended_summary_

        Parameters
        ----------
        pathMat : tacs.TACS.ContinuationPathMat
            _description_
        rhsVec : tacs.TACS.Vec
            _description_
        constraint : Optional[float], optional
            _description_, by default None
        """

    def computeForceVectors(self) -> None:
        """Compute the current forcing vector

        The continuation solver is based on the assumption that the residual takes the following form:

        r(u, lambda) = F_int(u) + lambda * F_ext(u, lambda)

        This function computes fInt and fExt using two residual evaluations at lambda=0 and lambda=1:
        f_int = r(u, 0)
        f_ext = r(u, 1) - f_int
        """
        currentLambda = self.getLambdaFunc()
        self.setLambdaFunc(0.0)
        self.resFunc(self.fInt)
        self.setLambdaFunc(1.0)
        self.resFunc(self.fExt)
        self.fExt.axpy(-1.0, self.fInt)
        self.setLambdaFunc(currentLambda)
        return

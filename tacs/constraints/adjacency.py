"""
The main purpose of this class is to constrain design variables step sizes across adjacent components.

.. note:: This class should be created using the
    :meth:`pyTACS.createAdjacencyConstraint <tacs.pytacs.pyTACS.createAdjacencyConstraint>` method.
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import scipy as sp

from tacs.constraints.base import TACSConstraint


class AdjacencyConstraint(TACSConstraint):
    def __init__(
        self,
        name,
        assembler,
        comm,
        outputViewer=None,
        meshLoader=None,
        options=None,
    ):
        """
        NOTE: This class should not be initialized directly by the user.
        Use pyTACS.createAdjacencyConstraint instead.

        Parameters
        ----------
        name : str
            Name of this tacs problem

        assembler : TACS.Assembler
            Cython object responsible for creating and setting tacs objects used to solve problem

        comm : mpi4py.MPI.Intracomm
            The comm object on which to create the pyTACS object.

        outputViewer : TACS.TACSToFH5
            Cython object used to write out f5 files that can be converted and used for postprocessing.

        meshLoader : pymeshloader.pyMeshLoader
            pyMeshLoader object used to create the assembler.

        options : dict
            Dictionary holding problem-specific option parameters (case-insensitive).
        """

        # Problem name
        self.name = name

        # Default setup for common constraint class objects, sets up comm and options
        TACSConstraint.__init__(
            self, assembler, comm, options, outputViewer, meshLoader
        )

        # Create a list of all adjacent components on root proc
        self._initializeAdjacencyList()

    def _initializeAdjacencyList(self):
        """
        Create a list of all components with common edges.
        """

        if self.comm.rank == 0:
            edgeToFace = {}
            nComp = self.meshLoader.getNumComponents()
            for compID in range(nComp):
                compConn = self.meshLoader.getConnectivityForComp(
                    compID, nastranOrdering=False
                )
                for elemConn in compConn:
                    nnodes = len(elemConn)
                    if nnodes >= 2:
                        for j in range(nnodes):
                            nodeID1 = elemConn[j]
                            nodeID2 = elemConn[(j + 1) % nnodes]

                            if nodeID1 < nodeID2:
                                key = (nodeID1, nodeID2)
                            else:
                                key = (nodeID2, nodeID1)

                            if key not in edgeToFace:
                                edgeToFace[key] = [compID]
                            elif compID not in edgeToFace[key]:
                                edgeToFace[key].append(compID)

            # Now we loop back over each element and each edge. By
            # using the edgeToFace dictionary, we can now determine
            # which components IDs (jComp) are connected to the
            # current component ID (iComp).
            self.adjacentComps = []

            for edgeKey in edgeToFace:
                if len(edgeToFace[edgeKey]) >= 2:
                    for i, iComp in enumerate(edgeToFace[edgeKey][:-1]):
                        for jComp in edgeToFace[edgeKey][i + 1 :]:
                            if iComp < jComp:
                                dvKey = (iComp, jComp)
                            else:
                                dvKey = (jComp, iComp)
                            if dvKey not in self.adjacentComps:
                                self.adjacentComps.append(dvKey)

        else:
            self.adjacentComps = None

        # Wait for root
        self.comm.barrier()

    def addConstraint(self, conName, lower=-1e20, upper=1e20, compIDs=None, dvIndex=0):
        """
        Generic method to adding a new constraint set for TACS.

        Parameters
        ----------
        conName : str
            The user-supplied name for the constraint set. This will
            typically be a string that is meaningful to the user

        lower: float or complex
            lower bound for constraint. Defaults to -1e20.

        upper: float or complex
            upper bound for constraint. Defaults to 1e20.

        compIDs: list or None
            List of compIDs to select. If None, all compIDs will be selected. Defaults to None.

        dvIndex : int
            Index number of element DV to be used in constraint. Defaults to 0.

        """
        if compIDs is not None:
            # Make sure CompIDs is flat and get element numbers on each proc corresponding to specified compIDs
            compIDs = self._flatten(compIDs)
        else:
            nComps = self.meshLoader.getNumComponents()
            compIDs = range(nComps)

        constrObj = self._createConstraint(dvIndex, compIDs, lower, upper)
        if constrObj.nCon > 0:
            self.constraintList[conName] = constrObj
            success = True
        else:
            self._TACSWarning(
                f"No adjacent components found in `compIDs`. Skipping {conName}."
            )
            success = False

        return success

    def _createConstraint(self, dvIndex, compIDs, lbound, ubound):
        size = self.comm.size
        rank = self.comm.rank
        nLocalDVs = self.getNumDesignVars()
        nLocalDVsOnProc = self.comm.gather(nLocalDVs, root=0)
        # Assemble constraint info on root proc
        if rank == 0:
            # Figure out which DVNums belong to each processor
            ownerRange = np.zeros(size + 1, dtype=int)
            # Sum local dv ranges over each proc to get global dv ranges
            ownerRange[1:] = np.cumsum(nLocalDVsOnProc)
            # Create a dict for converting global dv nums to local dv nums on each proc
            globalToLocalDVNumsOnProc = []
            for rank in range(size):
                g2ldv = dict(
                    zip(range(ownerRange[rank], ownerRange[rank + 1]), range(nLocalDVs))
                )
                globalToLocalDVNumsOnProc.append(g2ldv)
            # Create a list of lists that will hold the sparse data info on each proc
            rowsOnProc = [[] for _ in range(size)]
            colsOnProc = [[] for _ in range(size)]
            valsOnProc = [[] for _ in range(size)]
            conCount = 0
            # Loop through all adjacent component pairs
            for compPair in self.adjacentComps:
                # Check if they are in the user provided compIDs
                if compPair[0] in compIDs and compPair[1] in compIDs:
                    # We found a new constraint
                    for i, comp in enumerate(compPair):
                        # Get the TACS element object associated with this compID
                        elemObj = self.meshLoader.getElementObject(comp, 0)
                        elemIndex = 0
                        # Get the dvs owned by this element
                        globalDvNums = elemObj.getDesignVarNums(elemIndex)
                        # Check if specified dv num is owned by each proc
                        for proc_i in range(size):
                            globalToLocalDVNums = globalToLocalDVNumsOnProc[proc_i]
                            if globalDvNums[dvIndex] in globalToLocalDVNums:
                                localDVNum = globalDvNums[dvIndex]
                                rowsOnProc[proc_i].append(conCount)
                                colsOnProc[proc_i].append(localDVNum)
                                if i == 0:
                                    valsOnProc[proc_i].append(1.0)
                                else:
                                    valsOnProc[proc_i].append(-1.0)
                                break
                    conCount += 1

        else:
            rowsOnProc = None
            colsOnProc = None
            valsOnProc = None
            conCount = 0

        # Scatter local sparse indices/values to remaining procs
        rows = self.comm.scatter(rowsOnProc, root=0)
        cols = self.comm.scatter(colsOnProc, root=0)
        vals = self.comm.scatter(valsOnProc, root=0)
        conCount = self.comm.bcast(conCount, root=0)

        return SparseConstraint(
            self.comm, rows, cols, vals, conCount, nLocalDVs, lbound, ubound
        )

    def getConstraintBounds(self, bounds, evalCons=None, ignoreMissing=False):
        """
        Get bounds for constraints. The constraints corresponding to the strings in
        `evalCons` are evaluated and updated into the provided
        dictionary.

        Parameters
        ----------
        bounds : dict
            Dictionary into which the constraint bounds are saved.
            Bounds will be saved as a tuple: (lower, upper)
        evalCons : iterable object containing strings.
            If not none, use these constraints to evaluate.
        ignoreMissing : bool
            Flag to supress checking for a valid constraint. Please use
            this option with caution.

        Examples
        --------
        >>> funcs = {}
        >>> adjConstraint.getConstraintBounds(funcs, 'LE_SPAR')
        >>> funcs
        >>> # Result will look like (if AdjacencyConstraint has name of 'c1'):
        >>> # {'c1_LE_SPAR': (array([-1e20]), array([1e20]))}
        """
        # Check if user specified which constraints to output
        # Otherwise, output them all
        if evalCons is None:
            evalFuncs = self.constraintList
        else:
            userFuncs = sorted(list(evalCons))
            evalFuncs = {}
            for func in userFuncs:
                if func in self.constraintList:
                    evalFuncs[func] = self.constraintList[func]

        if not ignoreMissing:
            for f in evalFuncs:
                if f not in self.constraintList:
                    raise self._TACSError(
                        f"Supplied constraint '{f}' has not been added "
                        "using addConstraint()."
                    )

        # Loop through each requested constraint set
        for funcName in evalFuncs:
            key = f"{self.name}_{funcName}"
            bounds[key] = self.constraintList[funcName].getBounds()

    def evalConstraints(self, funcs, evalCons=None, ignoreMissing=False):
        """
        Evaluate values for constraints. The constraints corresponding to the strings in
        evalCons are evaluated and updated into the provided
        dictionary.

        Parameters
        ----------
        funcs : dict
            Dictionary into which the constraints are saved.
        evalCons : iterable object containing strings.
            If not none, use these constraints to evaluate.
        ignoreMissing : bool
            Flag to supress checking for a valid constraint. Please use
            this option with caution.

        Examples
        --------
        >>> funcs = {}
        >>> adjConstraint.evalConstraints(funcs, 'LE_SPAR')
        >>> funcs
        >>> # Result will look like (if AdjacencyConstraint has name of 'c1'):
        >>> # {'c1_LE_SPAR': array([12354.10])}
        """
        # Check if user specified which constraints to output
        # Otherwise, output them all
        if evalCons is None:
            evalCons = self.constraintList
        else:
            userCons = sorted(list(evalCons))
            evalCons = {}
            for func in userCons:
                if func in self.constraintList:
                    evalCons[func] = self.constraintList[func]

        if not ignoreMissing:
            for f in evalCons:
                if f not in self.constraintList:
                    raise self._TACSError(
                        f"Supplied constraint '{f}' has not been added "
                        "using addConstraint()."
                    )

        # Loop through each requested constraint set
        for conName in evalCons:
            key = f"{self.name}_{conName}"
            funcs[key] = self.constraintList[conName].evalCon(self.x.getArray())

    def evalConstraintsSens(self, funcsSens, evalCons=None):
        """
        This is the main routine for returning useful (sensitivity)
        information from constraint. The derivatives of the constraints
        corresponding to the strings in evalCons are evaluated and
        updated into the provided dictionary. The derivitives with
        respect to all design variables and node locations are computed.

        Parameters
        ----------
        funcsSens : dict
            Dictionary into which the derivatives are saved.
        evalCons : iterable object containing strings
            The constraints the user wants returned

        Examples
        --------
        >>> funcsSens = {}
        >>> adjConstraint.evalConstraintsSens(funcsSens, 'LE_SPAR')
        >>> funcsSens
        >>> # Result will look like (if AdjacencyConstraint has name of 'c1'):
        >>> # {'c1_LE_SPAR':{'struct':<50x242 sparse matrix of type '<class 'numpy.float64'>' with 100 stored elements in Compressed Sparse Row format>}}
        """
        # Check if user specified which functions to output
        # Otherwise, output them all
        if evalCons is None:
            evalCons = self.constraintList
        else:
            userCons = sorted(list(evalCons))
            evalCons = {}
            for con in userCons:
                if con in self.constraintList:
                    evalCons[con] = self.constraintList[con]

        # Get number of nodes coords on this proc
        nCoords = self.getNumCoordinates()

        # Loop through each requested constraint set
        for conName in evalCons:
            key = f"{self.name}_{conName}"
            # Get sparse Jacobian for dv sensitivity
            funcsSens[key] = {}
            funcsSens[key][self.varName] = self.constraintList[conName].evalConSens(
                self.x.getArray()
            )

            # Nodal sensitivities are always zero for this constraint,
            # Add an empty sparse matrix
            nCon = self.constraintList[conName].nCon
            funcsSens[key][self.coordName] = sp.sparse.csr_matrix(
                (nCon, nCoords), dtype=self.dtype
            )


class SparseConstraint(object):
    dtype = AdjacencyConstraint.dtype

    def __init__(self, comm, rows, cols, vals, nrows, ncols, lb=-1e20, ub=1e20):
        # Sparse Jacobian for constraint
        self.A = sp.sparse.csr_matrix(
            (vals, (rows, cols)), shape=(nrows, ncols), dtype=self.dtype
        )
        # Number of constraints
        self.nCon = nrows
        # MPI comm
        self.comm = comm
        # Save bound information
        if isinstance(lb, np.ndarray) and len(lb) == self.nCon:
            self.lb = lb.astype(self.dtype)
        elif isinstance(lb, float) or isinstance(lb, complex):
            self.lb = np.array([lb] * self.nCon, dtype=self.dtype)

        if isinstance(ub, np.ndarray) and len(ub) == self.nCon:
            self.ub = ub.astype(self.dtype)
        elif isinstance(ub, float) or isinstance(ub, complex):
            self.ub = np.array([ub] * self.nCon, dtype=self.dtype)

    def evalCon(self, x):
        conVals = self.comm.allreduce(self.A.dot(x))
        return conVals

    def evalConSens(self, x=None):
        return self.A.copy()

    def getBounds(self):
        return self.lb.copy(), self.ub.copy()

"""
==============================================================================
Panel Length Constraint
==============================================================================
@File    :   panel_length.py
@Date    :   2023/04/23
@Author  :   Alasdair Christison Gray
@Description : This class implements a constraint which enforces the panel
length design variable values passed to elements using the BladeStiffenedShell
constitutive model to be consistent with the true length of the panel they are
a part of.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
from jax import jit, jacrev
import jax.numpy as jnp
from jax.config import config
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs.constraints.base import TACSConstraint

config.update("jax_enable_x64", True)


@jit
def computePanelLength(points, direction):
    """Given the sorted points around the perimeter of a panel, compute the length of the panel in a given direction

    _extended_summary_

    Parameters
    ----------
    points : n x 3 jax array
        Coordinates of the perimeter points of the panel, in sorted order, so that points[i] - points[i-1] is a vector
        along the perimeter
    direction : length 3 jax array
        Direction in which to compute the panel length
    """

    numPoints = points.shape[0]
    normalisedDirection = direction / jnp.linalg.norm(direction)

    # --- Find the "average plane" of the panel by computing an SVD basis of the perimiter points ---
    # subtract out the centroid and take the SVD
    centroid = jnp.mean(points, axis=0, keepdims=True)
    centredPoints = points - centroid
    _, _, VT = jnp.linalg.svd(centredPoints, full_matrices=False)

    # The first two right singular vectors are the basis vectors of the plane
    planeBasis = VT[:2]

    # --- Project the points and direction onto the plane ---
    projectedPoints = centredPoints @ planeBasis.T
    projectedDirection = normalisedDirection @ planeBasis.T
    projectDirectionScale = jnp.linalg.norm(projectedDirection)
    projectedDirection = projectedDirection / projectDirectionScale

    # --- Intersection computation ---
    # For each point in the perimeter, we now compute the point at which a line in the length direction starting at the
    # point intersects the line along each edge in the panel. If that intersection occures within the bounds of the
    # edge, we compute the length of the line from the current point to the intersection and this becomes a candidate
    # value for the panel length
    length = 0.0
    for pointInd in range(numPoints):
        for edgeInd in range(numPoints):
            startInd = edgeInd
            endInd = (edgeInd + 1) % numPoints
            # We only need to check edges that are not adjacent to the current point
            if not (startInd == pointInd or endInd == pointInd):
                edge = projectedPoints[endInd] - projectedPoints[startInd]
                # Solve the equation projectedPoints[pointInd] + alpha * projectedDirection = projectedPoints[startInd] + beta * edge to find the intersection
                rhs = projectedPoints[startInd] - projectedPoints[pointInd]
                A = jnp.stack([projectedDirection, -edge], axis=1)
                sol = jnp.linalg.solve(A, rhs)
                beta = sol[1]
                if beta <= 1 and beta >= 0:
                    # The intersection is within the bounds of the edge, so compute the length of the line
                    # from the current point to the intersection back in 3D space
                    startPoint = points[pointInd]
                    endPoint = points[startInd] + beta * (
                        points[endInd] - points[startInd]
                    )
                    newLength = jnp.linalg.norm(endPoint - startPoint)
                    length = jnp.maximum(length, newLength)
    return length


computePanelLengthSens = jit(jacrev(computePanelLength, argnums=0))


def simplifyPoly(nodeIDs, nodes, angleTol=18.0):
    """
    Take a (closed) chain of nodes and remove any nodes that turn by less than angleTol degrees.
    This simplifies the polygon by leaving only "sharp" corners
    """
    cont = True
    while cont:
        newNodes = []
        newNodeIDs = []
        for i in range(len(nodes)):
            im1 = i - 1
            ip1 = i + 1
            if i == 0:
                im1 = len(nodes) - 1
            if i == len(nodes) - 1:
                ip1 = 0

            v1 = nodes[ip1] - nodes[i]
            v2 = nodes[im1] - nodes[i]
            # Angle between vectors
            arg = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            arg = max(-1, arg)
            arg = min(1, arg)
            theta = np.arccos(arg)

            if theta < np.pi * (1 - angleTol / 180):
                newNodes.append(nodes[i])
                newNodeIDs.append(nodeIDs[i])

        if len(newNodes) == len(nodes):
            cont = False

        nodes = np.array(newNodes)
        nodeIDs = np.array(newNodeIDs)

    return list(nodeIDs), nodes


class PanelLengthConstraint(TACSConstraint):
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

        # Dictionary for storing the IDs of the nodes on the boundary of each component that is subject to a constraint
        self.boundaryNodeIDs = {}
        self.compIDs = []
        self.dvInds = []

    def addConstraint(self, conName, compIDs=None, lower=None, upper=None, dvIndex=0):
        """
        Generic method to adding a new constraint set for TACS.

        Parameters
        ----------
        conName : str
            The user-supplied name for the constraint set. This will
            typically be a string that is meaningful to the user

        compIDs: list[int] or None
            List of compIDs to select. If None, all compIDs will be selected. Defaults to None.

        lower: float or complex
            lower bound for constraint. Not used.

        upper: float or complex
            upper bound for constraint. Not used.

        dvIndex : int
            Index number of the panel length DV's. Defaults to 0.

        """
        if compIDs is not None:
            # Make sure CompIDs is flat and get element numbers on each proc corresponding to specified compIDs
            compIDs = self._flatten(compIDs)
        else:
            nComps = self.meshLoader.getNumComponents()
            compIDs = range(nComps)

        # Get the boundary node IDs for each component
        boundaryNodeIDs = self._getComponentBoundaryNodes(compIDs)

        # Now figure out which proc is in charge of the DV's for each component
        compIDs = []
        dvInds = []
        boundaryNodeIDs = []
        refAxes = []
        for compID in compIDs:
            # Get the TACS element object associated with this compID
            elemObj = self.meshLoader.getElementObject(compID, 0)
            transObj = elemObj.getTransform()
            refAxis = transObj.getRefAxis()
            globalDvNums = elemObj.getDesignVarNums(0)
            globalDVNum = globalDvNums[dvIndex]
            if globalDVNum in self.globalToLocalDVNums:
                # This proc is in charge of this DV, so store the necessary info on this proc
                compIDs.append(compID)
                dvInds.append(self.globalToLocalDVNums[globalDVNum])
                boundaryNodeIDs.append(boundaryNodeIDs[compID])
                refAxes.append(refAxis)

        # Add the constraint to the constraint list
        if len(compIDs) != 0:
            self.constraintList[conName] = {
                "compIDs": compIDs,
                "dvInds": dvInds,
                "boundaryNodeIDs": boundaryNodeIDs,
                "refAxes": refAxes,
            }
        else:
            self.constraintList[conName] = None

        # TODO: Need to add something here to keep track of global size of constraint array and which entries are on which proc?

        success = True

        return success

    def evalConstraints(self, funcs, evalCons=None, ignoreMissing=False):
        # Check if user specified which constraints to output
        # Otherwise, output them all
        evalCons = self._processEvalCons(evalCons, ignoreMissing)
        # Loop through each requested constraint set
        for conName in evalCons:
            if self.constraintList[conName] is not None:
                key = f"{self.name}_{conName}"

    def _getComponentBoundaryNodes(self, compIDs):
        """For a given list of components, find the nodes on the boundaries of each of the components.

        The nodeIDs computed here are broadcast to all procs

        Parameters
        ----------
        compIDs: list[int]
            List of compIDs to find boundaries of.

        Returns
        --------
        dict[int, list[int]]
            Dictionary where dict[compID] = sorted list of nodeIDs on the boundary of the component
        """
        # Make sure CompIDs is flat
        compIDs = self._flatten(compIDs)

        boundaryNodeIDs = {}

        if self.rank == 0:
            for compID in compIDs:
                allEdges = set()
                dupEdges = set()
                compConn = self.meshLoader.getConnectivityForComp(
                    compID, nastranOrdering=False
                )
                # Go over all the elements in the component and add their edges to the sets of all and possibly duplicate edges
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

                            # Skip degenerate edges
                            if key[0] != key[1]:
                                # Either add to allEdges or dupEdges depending on whether we've seen this edge before
                                if key not in allEdges:
                                    allEdges.add(key)
                                else:
                                    dupEdges.add(key)
                # Now get a list of all the edges that aren't duplicated, these are the boundary edges
                boundaryEdges = list(allEdges - dupEdges)

                # Create a nodeToElem Pointer using a dictionary:
                nodeToElem = {}
                for iEdge in range(len(boundaryEdges)):
                    edge = boundaryEdges[iEdge]
                    for ii in range(2):
                        if edge[ii] in nodeToElem:
                            nodeToElem[edge[ii]].append(iEdge)
                        else:
                            nodeToElem[edge[ii]] = [iEdge]

                # Now check that each nodeToElem has a length of
                # 2. This means we have a chance it is a
                # closed curve
                for key in nodeToElem:
                    if len(nodeToElem[key]) != 2:
                        raise ValueError(
                            "The topology of the geometry associated with "
                            "a constitutive object is not manifold "
                            "(There is a node with three or more edges "
                            "attached. This constitutive object cannot "
                            "use a panel-type constitutive object. "
                            f"CompIDs are: {repr(compIDs)}"
                        )

                # Now we will "order" the edges if possible. This
                # will also allow us to detect multiple loops
                # which isn't allowed, or a non-manifold local
                # geometry - ie. in this context a node connected
                # to three edges. This is also not allowed.

                nodeChain = [boundaryEdges[0][0], boundaryEdges[0][1]]
                cont = True
                curElem = 0
                while cont:
                    # We arbitrarily pick the first 'element'
                    # (edge) containing the first two nodes of our
                    # chain. Next step is to find the next element
                    # and node in the chain:
                    nextElems = nodeToElem[nodeChain[-1]]
                    # Get the 'nextElem' that isn't the current
                    # one
                    if nextElems[0] == curElem:
                        nextElem = nextElems[1]
                    else:
                        nextElem = nextElems[0]

                    # Now nextElem is the next in the chain. Get
                    # the nodes for this elem:
                    nextNodes = boundaryEdges[nextElem]

                    # Append the node that isn't the last one
                    # (that is already in the chain)
                    if nextNodes[0] == nodeChain[-1]:
                        nodeChain.append(nextNodes[1])
                    else:
                        nodeChain.append(nextNodes[0])

                    # Exit condition:
                    if nodeChain[-1] == nodeChain[0]:
                        # We've made it all the way around!
                        cont = False

                    # Set current element
                    curElem = nextElem

                # Now check that we've *actually* used all of our
                # nodes. Since we've determined it is manifold,
                # this must mean we have multiple loops which
                # *also* isn't allowed.
                if len(nodeChain) - 1 != len(boundaryEdges):
                    raise ValueError(
                        "Detected more than one closed loop for "
                        "constitutive object. This is not allowed. "
                        "This constitutive object cannot use a "
                        "panel-type constitutive object. "
                        f"CompIDs are: {repr(compIDs)}"
                    )
                nodeChain = nodeChain[:-1]
                nodeChainCoords = self.meshLoader.getBDFNodes(
                    nodeChain, nastranOrdering=False
                )
                nodeIDs, coords = simplifyPoly(nodeChain, nodeChainCoords)
                boundaryNodeIDs[compID] = nodeIDs

        return self.comm.bcast(boundaryNodeIDs, root=0)

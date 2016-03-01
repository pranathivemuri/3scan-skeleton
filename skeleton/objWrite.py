import numpy as np
import networkx as nx
import itertools

import time

from skeleton.networkxGraphFromarray import getNetworkxGraphFromarray
from skeleton.radiusOfNodes import getRadiusByPointsOnCenterline
from skeleton.segmentLengths import _removeEdgesInVisitedPath


"""
    write an array to a wavefront obj file - time takes upto 3 minutes for a 512 * 512 * 512 array,
    input array can be either 3D or 2D. Function needs to be called with an array you want to save
    as a .obj file and the location on obj file
"""


def getObjWrite(imArray, pathTosave):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray, True)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
    else:
        networkxGraph = imArray
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices
    strsVertices = []
    listVertex = list(map(list, verticesSorted))
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
        vertex = tuple((listVertex[index][0], listVertex[index][1] * 0.6, listVertex[index][2] * 0.6))
        strsVertices.append("v " + " ".join(str(vertex[i - 2]) for i in range(0, len(vertex))) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    strsSeq = []
    # line prefixes for the obj file
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            continue
        """ if there are more than one nodes decide what kind of subgraph it is
            if it has cycles alone, or is a cyclic graph with a tree or an
            acyclic graph with tree """
        nodes.sort()
        cycleList = nx.cycle_basis(subGraphskeleton)
        cycleList = [item for item in cycleList if len(item) != 3]
        cycleCount = len(cycleList)
        nodeDegreedict = nx.degree(subGraphskeleton)
        degreeList = list(nodeDegreedict.values())
        endPointdegree = min(degreeList)
        branchPointdegree = max(degreeList)
        if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount != 0:
            """ if the maximum degree is equal to minimum degree it is a circle"""
            cycle = cycleList[0]
            cycle.append(cycle[0])
            strsSeq.append("l " + " ".join(str(x) for x in cycle) + "\n")
            _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
        elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
            branchpoints = nodes
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            modulus = [[start - end] for start, end in listOfPerms]
            dists = [abs(i[0]) for i in modulus]
            if len(list(nx.articulation_points(subGraphskeleton))) == 1 and set(dists) != 1:
                """ each node is connected to one or two other nodes implies and there is a
                    one branch point at a distance not equal to one it is a single dichotomous tree"""
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if len(list(set(branchpoints) & set(simplePath))) == 2:
                                strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            else:
                """ each node is connected to one or two other nodes implies it is a line,
                set tortuosity to 1"""
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
                strsSeq.append("l " + " ".join(str(x) for x in nodes) + "\n")
        elif cycleCount >= 1:
            """go through each of the cycles"""
            for nthCycle, cyclePath in enumerate(cycleList):
                _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                cyclePath.append(cyclePath[0])
                strsSeq.append("l " + " ".join(str(x) for x in cyclePath) + "\n")
            "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
            # collecting all the branch and endpoints
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
            branchpoints.sort()
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        if len(list(set(branchpoints) & set(simplePath))) == 2:
                            strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        else:
            "acyclic tree characteristics"""
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
            branchpoints.sort()
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        if len(list(set(branchpoints) & set(simplePath))) == 2:
                            strsSeq.append("l " + " ".join(str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        assert subGraphskeleton.number_of_edges() == 0
    objFile.writelines(strsSeq)
    print("obj file write took %0.3f seconds" % (time.time() - startt))
    # Close opend file
    objFile.close()


def getObjWriteWithradius(imArray, pathTosave, dictOfNodesAndRadius):
    """
       takes in a numpy array and converts it to a obj file and writes it to pathTosave
    """
    startt = time.time()  # for calculating time taken to write to an obj file
    if type(imArray) == np.ndarray:
        networkxGraph = getNetworkxGraphFromarray(imArray, True)  # converts array to a networkx graph(based on non zero coordinates and the adjacent nonzeros)
    else:
        networkxGraph = imArray
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    GraphList = nx.to_dict_of_lists(networkxGraph)  # convert the graph to a dictionary with keys as nodes and list of adjacent nodes as the values
    verticesSorted = list(GraphList.keys())  # list and sort the keys so they are geometrically in the same order when writing to an obj file as (l_prefix paths)
    verticesSorted.sort()
    mapping = {}  # initialize variables for writing string of vertex v followed by x, y, x coordinates in the obj file
    #  for each of the sorted vertices write both v and vy followed by radius
    strsVertices = [0] * (2 * len(verticesSorted))
    for index, vertex in enumerate(verticesSorted):
        mapping[vertex] = index + 1  # a mapping to transform the vertices (x, y, z) to indexes (beginining with 1)
        originalVertex = list(vertex)
        newVertex = [0] * len(vertex)
        newVertex[0] = originalVertex[0] / 1; newVertex[1] = originalVertex[2] * 0.6; newVertex[2] = originalVertex[1] * 0.6; vertex = tuple(newVertex)
        strsVertices[index] = "v " + " ".join(str(vertex[i - 2]) for i in range(0, len(vertex))) + "\n"  # add strings of vertices to obj file
        strsVertices[index + len(verticesSorted)] = "vt " + " " + str(dictOfNodesAndRadius[vertex]) + "\n"
    objFile.writelines(strsVertices)  # write strings to obj file
    networkGraphIntegerNodes = nx.relabel_nodes(networkxGraph, mapping, False)
    strsSeq = []
    # line prefixes for the obj file
    disjointGraphs = list(nx.connected_component_subgraphs(networkGraphIntegerNodes))
    for ithDisjointgraph, subGraphskeleton in enumerate(disjointGraphs):
        nodes = subGraphskeleton.nodes()
        if len(nodes) == 1:
            continue
        """ if there are more than one nodes decide what kind of subgraph it is
            if it has cycles alone, or is a cyclic graph with a tree or an
            acyclic graph with tree """
        nodes.sort()
        cycleList = nx.cycle_basis(subGraphskeleton)
        cycleList = [item for item in cycleList if len(item) != 3]
        cycleCount = len(cycleList)
        nodeDegreedict = nx.degree(subGraphskeleton)
        degreeList = list(nodeDegreedict.values())
        endPointdegree = min(degreeList)
        branchPointdegree = max(degreeList)
        if endPointdegree == branchPointdegree and nx.is_biconnected(subGraphskeleton) and cycleCount != 0:
            """ if the maximum degree is equal to minimum degree it is a circle"""
            cycle = cycleList[0]
            cycle.append(cycle[0])
            strsSeq.append("l " + " ".join(str(x) + "/" + str(x) for x in cycle) + "\n")
            _removeEdgesInVisitedPath(subGraphskeleton, cycle, 1)
        elif set(degreeList) == set((1, 2)) or set(degreeList) == {1}:
            branchpoints = nodes
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            modulus = [[start - end] for start, end in listOfPerms]
            dists = [abs(i[0]) for i in modulus]
            if len(list(nx.articulation_points(subGraphskeleton))) == 1 and set(dists) != 1:
                """ each node is connected to one or two other nodes implies and there is a
                    one branch point at a distance not equal to one it is a single dichotomous tree"""
                for sourceOnTree, item in listOfPerms:
                    if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                        simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                        for simplePath in simplePaths:
                            if len(list(set(branchpoints) & set(simplePath))) == 2:
                                strsSeq.append("l " + " ".join(str(x) + "/" + str(x) for x in simplePath) + "\n")
                                _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
            else:
                """ each node is connected to one or two other nodes implies it is a line,
                set tortuosity to 1"""
                edges = subGraphskeleton.edges()
                subGraphskeleton.remove_edges_from(edges)
                strsSeq.append("l " + " ".join(str(x) + "/" + str(x) for x in nodes) + "\n")
        elif cycleCount >= 1:
            """go through each of the cycles"""
            for nthCycle, cyclePath in enumerate(cycleList):
                _removeEdgesInVisitedPath(subGraphskeleton, cyclePath, 1)
                cyclePath.append(cyclePath[0])
                strsSeq.append("l " + " ".join(str(x) for x in cyclePath) + "\n")
            "all the cycles in the graph are checked now look for the tree characteristics in this subgraph"
            # collecting all the branch and endpoints
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
            branchpoints.sort()
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        if len(list(set(branchpoints) & set(simplePath))) == 2:
                            strsSeq.append("l " + " ".join(str(x) + "/" + str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        else:
            "acyclic tree characteristics"""
            branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2]
            branchpoints.sort()
            listOfPerms = list(itertools.permutations(branchpoints, 2))
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(subGraphskeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(subGraphskeleton, source=sourceOnTree, target=item))
                    for simplePath in simplePaths:
                        if len(list(set(branchpoints) & set(simplePath))) == 2:
                            strsSeq.append("l " + " ".join(str(x) + "/" + str(x) for x in simplePath) + "\n")
                            _removeEdgesInVisitedPath(subGraphskeleton, simplePath, 0)
        assert subGraphskeleton.number_of_edges() == 0
    objFile.writelines(strsSeq)
    print("obj file write took %0.3f seconds" % (time.time() - startt))
    # Close opend file
    objFile.close()


if __name__ == '__main__':
    # read points into array
    skeletonIm = np.load(input("enter a path to shortest path skeleton volume------"))
    boundaryIm = np.load(input("enter a path to boundary of thresholded volume------"))
    dictOfNodesAndRadius, distTransformedIm = getRadiusByPointsOnCenterline(skeletonIm, boundaryIm)
    getObjWriteWithradius(skeletonIm, "PV_rT.obj", dictOfNodesAndRadius)
    # truthCase = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton.npy")
    # groundTruth = np.load("/home/pranathi/Downloads/twodimageslices/output/Skeleton-gt.npy")
    # getObjWrite(truthCase, "PV_T.obj")
    # getObjWrite(groundTruth, "PV_GT.obj")

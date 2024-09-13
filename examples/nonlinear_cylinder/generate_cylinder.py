from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pprint import pprint
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nex",
    type=int,
    default=16,
    help="# elements along the circumference of the cylinder",
)
parser.add_argument(
    "--ney", type=int, default=16, help="# elements along the length of the cylinder"
)
args = parser.parse_args()

# Overall dimensions
radius = 2.540
length = radius / 10
angle = 0.1  # radians

# Number of elements along each edge of a single panel
nex = args.nex
ney = args.ney

np1 = 1
np2 = 1

# Nodes
n1 = nex + 1
n2 = ney + 1
theta = np.linspace(0, angle, n1)
y = np.linspace(0, length, n2)
THETA, Y = np.meshgrid(theta, y)
X = radius * np.sin(THETA)
Z = radius * np.cos(THETA)

# Node numbering
nid = np.zeros((n2, n1), dtype="intc")
bcnodes = []
count = 1
for i in range(n2):
    isYmin = i == 0
    isYmax = i == n2 - 1
    for j in range(n1):
        isXmin = j == 0
        isXmax = j == n1 - 1
        nid[i, j] = count
        fixedDOF = []
        # XMin edge is a x symmetry plane, fixed in x translation and y/z rotation
        if isXmin:
            fixedDOF += [1, 5, 6]
        # XMax edge is hinged
        if isXmax:
            fixedDOF += [1, 2, 3]
        # YMin edge is a y symmetry plane, fixed in y translation and x/z rotation
        if isYmin:
            fixedDOF += [2, 4, 6]
        if len(fixedDOF) > 0:
            # Get the unique fixed DOF in ascending order
            fixedDOF = list(set((sorted(fixedDOF))))
            bcnodes.append({"nodenum": count, "fixedDOF": "".join(map(str, fixedDOF))})
        count += 1
nodes = np.stack((X, Y, Z), axis=2)
nmat = nodes.reshape((n1 * n2, 3))

# Connectivity
nex = n1 - 1
ney = n2 - 1
ne = nex * ney
ncomp = 1
conn = {i + 1: [] for i in range(ncomp)}
ie = 1
for i in range(ney):
    for j in range(nex):
        compID = i // ney * np1 + j // nex + 1
        conn[compID].append(
            [ie, nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]]
        )
        ie += 1


# Write BDF
output_file = "cylinder.bdf"
fout = open(output_file, "w")


def write_80(line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)


write_80("SOL 103")
write_80("CEND")
write_80("BEGIN BULK")

# Make component names
compNames = {}
compID = 1
for i in range(np2):
    for j in range(np1):
        compNames[compID] = "PLATE.{:03d}/SEG.{:02d}".format(i, j)
        compID += 1


def write_bulk_line(key, items, format="small"):
    if format == "small":
        width = 8
        writekey = key
    elif format == "large":
        width = 16
        writekey = key + "*"
    line = "{:8s}".format(writekey)
    for item in items:
        if type(item) in [int, np.int64, np.int32]:
            line += "{:{width}d}".format(item, width=width)[:width]
        elif type(item) in [float, np.float64]:
            line += "{: {width}f}".format(item, width=width)[:width]
        elif type(item) is str:
            line += "{:{width}s}".format(item, width=width)[:width]
        else:
            print(type(item), item)
        if len(line) == 72:
            write_80(line)
            line = " " * 8
    if len(line) > 8:
        write_80(line)


# Write nodes
for i in range(n1 * n2):
    write_bulk_line("GRID", [i + 1, 0, nmat[i, 0], nmat[i, 1], nmat[i, 2], 0, 0, 0])

# Write elements
compID = 1
for key in conn:
    famPrefix = "$       Shell element data for family    "
    famString = "{}{:39s}".format(famPrefix, compNames[compID])
    write_80(famString)
    compID += 1
    for element in conn[key]:
        element.insert(1, key)
        write_bulk_line("CQUAD4", element)

# Write boundary conditions
for node in bcnodes:
    write_bulk_line("SPC", [1, node["nodenum"], node["fixedDOF"], 0.0])

write_80("ENDDATA")

fout.close()

"""
Utilities for reconstructing valid cuboid wireframe topology from 8
unordered/ambiguous 3D corner points, since CEPB's raw vertex ordering
cannot be assumed consistent across objects.
"""

import numpy as np
from itertools import combinations


def get_cuboid_edges(cuboid_3d, face_a=(0, 1, 4, 5), face_b=(2, 3, 6, 7)):
    """
    Reconstruct the 12 edges of a box from 8 3D corner points.

    Splits corners into two face groups, finds the shortest 4-cycle within
    each face (2 edges per vertex, greedy shortest-first), then connects
    each face_a vertex to its nearest face_b counterpart.

    Args:
        cuboid_3d: (8, 3) array of corner points.
        face_a, face_b: index groupings of the two opposite faces.

    Returns:
        List of 12 (i, j) index tuples representing box edges.
    """
    cuboid_3d = np.asarray(cuboid_3d)
    dist = np.linalg.norm(cuboid_3d[:, None] - cuboid_3d[None, :], axis=2)

    def face_cycle(face):
        pairs = list(combinations(face, 2))
        pairs.sort(key=lambda p: dist[p[0], p[1]])
        edges, deg = [], {v: 0 for v in face}
        for i, j in pairs:
            if deg[i] < 2 and deg[j] < 2:
                edges.append((i, j))
                deg[i] += 1
                deg[j] += 1
            if len(edges) == 4:
                break
        return edges

    edges = face_cycle(face_a) + face_cycle(face_b)
    for i in face_a:
        j = min(face_b, key=lambda x: dist[i, x])
        edges.append((i, j))

    return sorted(set(tuple(sorted(e)) for e in edges))

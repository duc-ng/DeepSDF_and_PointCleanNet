import trimesh
import numpy as np
import numpy as np
from scipy.spatial import KDTree


def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)


# init
nr_samples = 30000
mesh_files = [
    "out/1_preprocessed/bunny.obj",
    "out/2_reconstructed/bunny.obj",
    "out/3_cleaned_outliers/bunny.obj",
    "out/4_cleaned_noise/bunny_2.obj",
]

# load meshes
meshes = [trimesh.load(f) for f in mesh_files]

# sample points
points = [trimesh.sample.sample_surface(m, nr_samples)[0] for m in meshes]

# compute the Chamfer Distance
distances = [chamfer_distance(points[0], points[i]) for i in range(1, len(points))]

print("Chamfer Distance (original and reconstructed):", distances[0])
print("Chamfer Distance (original and cleaned outliers):", distances[1])
print("Chamfer Distance (original and cleaned noise):", distances[2])

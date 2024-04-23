import os
import trimesh
import argparse
from glob import glob
import numpy as np
from os.path import basename, join

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess .obj files")
    parser.add_argument("--input_dir", type=str, default="data", help="Input directory")
    parser.add_argument(
        "--output_dir", type=str, default="out/1_preprocessed", help="Output directory"
    )
    parser.add_argument(
        "--samples_on_surface",
        type=int,
        default=15000,
        help="Number of samples on the surface",
    )
    parser.add_argument(
        "--samples_in_bbox",
        type=int,
        default=15000,
        help="Number of samples in the bounding box",
    )
    return parser.parse_args()


def save_samples(samples_and_sdfs: np.ndarray, out_file: str):
    """
    Save samples and SDFs to a .npz file.
    """
    sdfs = samples_and_sdfs[:, 3]
    pos = samples_and_sdfs[sdfs > 0].reshape(-1, 4)
    neg = samples_and_sdfs[sdfs <= 0].reshape(-1, 4)
    np.savez(out_file, pos=pos, neg=neg)


def generate_samples(mesh_file: str, samples_in_bbox: int, samples_on_surface: int):
    """
    Generate samples and signed distance functions (SDFs) for a mesh.
    """
    # load mesh
    mesh = trimesh.load_mesh(mesh_file)
    assert mesh.vertices.shape[1] == 3, f"Vertices of {mesh_file} are not 3D"

    # normalize to [-1, 1] and center
    mesh.apply_scale(2 / np.max(mesh.extents))
    mesh.apply_translation(-mesh.centroid)
    
    # save normalized mesh
    mesh.export(join(args.output_dir, basename(mesh_file)))

    # generate samples
    v_min, v_max = mesh.vertices.min(0), mesh.vertices.max(0)
    samples_bbox = np.random.uniform(v_min, v_max, (samples_in_bbox, 3))
    samples_surface, _ = trimesh.sample.sample_surface(mesh, samples_on_surface)

    # compute signed distance
    sdf_points = mesh.nearest.signed_distance(samples_bbox)
    sdf_surface = np.zeros(samples_surface.shape[0])

    # stack values
    sdfs = np.hstack((sdf_points, sdf_surface))  # (n,)
    samples = np.vstack((samples_bbox, samples_surface))  # (n,3)
    samples_and_sdfs = np.hstack((samples, sdfs.reshape(-1, 1)))  # (n,4)
    return samples_and_sdfs


if __name__ == "__main__":
    """
    Preprocess .obj files by generating samples and signed distance functions (SDFs).
    """
    # init
    args = parse_args()
    mesh_files = glob(join(args.input_dir, "*.obj"))
    nr_files = len(mesh_files)
    os.makedirs(args.output_dir, exist_ok=True)

    # process files
    print(f"Preprocessing {nr_files} files...")
    for i, mesh_file in enumerate(mesh_files):

        # generate samples
        print(f"{i+1}/{nr_files} Preprocessing", mesh_file)
        samples_and_sdfs = generate_samples(
            mesh_file,
            args.samples_in_bbox,
            args.samples_on_surface,
        )

        # save samples and SDFs
        out_file = join(args.output_dir, basename(mesh_file).replace(".obj", ".npz"))
        save_samples(samples_and_sdfs, out_file)
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
        default=10000,
        help="Number of samples on the surface",
    )
    parser.add_argument(
        "--samples_in_bbox",
        type=int,
        default=10000,
        help="Number of samples in the bounding box",
    )
    parser.add_argument(
        "--samples_in_cube",
        type=int,
        default=5000,
        help="Number of samples in the cube",
    )
    return parser.parse_args()




def generate_samples(mesh_file: str, samples_in_bbox: int, samples_on_surface: int, samples_in_cube: int):
    """
    Generate samples and signed distance functions (SDFs) for a mesh.
    """
    # load mesh
    mesh = trimesh.load_mesh(mesh_file)
    assert mesh.vertices.shape[1] == 3, f"Vertices of {mesh_file} are not 3D"

    # normalize to [-1, 1] with 3% margin
    mesh.apply_scale(2 / ((mesh.bounds[1]-mesh.bounds[0]).max() * 1.03))
    mesh.apply_translation(-(mesh.bounds[1] + mesh.bounds[0]) / 2)    
    # save normalized mesh
    mesh.export(join(args.output_dir, basename(mesh_file)))

    # generate samples
    v_min, v_max = mesh.vertices.min(0), mesh.vertices.max(0)
    samples_bbox = np.random.uniform(v_min, v_max, (samples_in_bbox, 3))
    samples_cube = np.random.rand(samples_in_cube, 3) * 2 - 1
    samples_surface, _ = trimesh.sample.sample_surface(mesh, samples_on_surface)
    samples = np.vstack((samples_bbox, samples_cube, samples_surface))  # (n,3)

    # compute signed distance 
    sdf_bbox = mesh.nearest.signed_distance(samples_bbox) * -1 
    sdf_cube = mesh.nearest.signed_distance(samples_cube) * -1
    sdf_surface = np.zeros(samples_surface.shape[0])
    sdf = np.hstack((sdf_bbox, sdf_cube, sdf_surface)) # (n,)

    return samples, sdf


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
        samples, sdf = generate_samples(
            mesh_file,
            args.samples_in_bbox,
            args.samples_on_surface,
            args.samples_in_cube
        )

        # save samples and SDFs
        out_file = join(args.output_dir, basename(mesh_file).replace(".obj", ".npz"))
        np.savez(out_file, samples=samples, sdf=sdf)
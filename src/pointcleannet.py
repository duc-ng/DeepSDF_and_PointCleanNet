import argparse
import os
import subprocess
from glob import glob
import numpy as np
import open3d as o3d
import trimesh


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Predict 3D shapes")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--input_dir", type=str, default="./out/2_reconstructed", help="Input directory"
    )
    parser.add_argument(
        "--outlier_dir",
        type=str,
        default="./out/3_cleaned_outliers",
        help="Directory to store files with outliers removed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out/4_cleaned_noise",
        help="Output directory",
    )
    parser.add_argument(
        "--nrun",
        type=int,
        default=2,
        help="Number of runs for noise removal",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def save_filenames(input_dir, filename, ending="obj"):
    """
    Save file names in .txt
    """
    obj_files = glob(os.path.join(input_dir, f"*.{ending}"))
    output_file = os.path.join(input_dir, filename)
    with open(output_file, "w") as file:
        for obj_file in obj_files:
            file_name_without_extension = os.path.splitext(os.path.basename(obj_file))[
                0
            ]
            file.write(file_name_without_extension + "\n")


def remove_outliers_from_info_file(input_dir, mesh_dir):
    """
    Remove outliers from .info file (x,y,z,pred) and save as .obj and .xyz file
    if  pred is close to outlier, remove point
    """
    info_files = glob(os.path.join(input_dir, "*.info"))
    for i, info_file in enumerate(info_files):
        # open  mesh
        base = os.path.basename(info_file).split(".")[0]
        mesh = trimesh.load_mesh(os.path.join(mesh_dir, f"{base}.obj"), process=False)
        
        # remove outliers from mesh 
        info = np.loadtxt(info_file)
        print("Number of vertices in mesh:", mesh.vertices.shape[0])
        print("Number of entries in mask:", info.shape[0])
        mask = info[:, 3] > 0.5
        new_vertices = mesh.vertices[mask] # remove outliers
                
        # save mesh as .obj
        mesh = trimesh.Trimesh(vertices=new_vertices)
        mesh.export(os.path.join(input_dir, f"{base}.obj"))
        
        # save vertices as .xyz        
        xyz_file = os.path.join(input_dir, f"{base}.xyz")
        np.savetxt(xyz_file, new_vertices, fmt='%.6f', delimiter=' ')            
        print(f"Removed outliers from {base}.info")

def remove_noise(directory, directory_mesh, nrun):
    """
    Remove noise from mesh and save as .obj
    """
    xyz_files = glob(os.path.join(directory, f"*_{nrun}.xyz"))
    for xyz_file in xyz_files:
        # open mesh 
        base = os.path.basename(xyz_file).split(".")[0].split("_")[0]
        mesh = trimesh.load_mesh(os.path.join(directory_mesh, f"{base}.obj"), process=False)
        
        # open .xyz file
        xyz = np.loadtxt(xyz_file)
        mesh.vertices = xyz
        
        # save mesh as .obj
        mesh.export(os.path.join(directory, f"{base}_{nrun}.obj"))


if __name__ == "__main__":
    # initialize
    config = parse_args()
    os.makedirs(config.outlier_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(config.seed)
    print("Using device (PointCleanNet):", config.device)

    # 1. save file names in validationset.txt
    save_filenames(config.input_dir, "validationset.txt")

    # 2. get outliers with PointCleanNet
    directory_path = "pointcleannet/outliers_removal"
    command = [
        "python",
        "eval_pcpnet.py",
        "--indir",
        os.path.abspath(config.input_dir),
        "--outdir",
        os.path.abspath(config.outlier_dir),
        "--device",
        config.device,
        "--seed",
        str(config.seed),
    ]
    subprocess.run(command, cwd=directory_path)

    # 3. remove outliers and save 
    remove_outliers_from_info_file(config.outlier_dir, config.input_dir)

    # 4. save file names in testset.txt
    save_filenames(config.outlier_dir, "testset.txt")

    # 5. remove noise 
    directory_path2 = "pointcleannet/noise_removal"
    basenames = [
        os.path.basename(f).split(".")[0]
        for f in glob(os.path.join(config.outlier_dir, "*.xyz"))
    ]
    for i in range(config.nrun):
        for basename in basenames:
            command2 = [
                "python",
                "eval_pcpnet.py",
                "--indir",
                os.path.abspath(config.outlier_dir),
                "--outdir",
                os.path.abspath(config.output_dir),
                "--device",
                config.device,
                "--nrun",
                str(i + 1),
                "--shapename",
                basename + "_{i}",
                "--seed",
                str(config.seed),
            ]
            subprocess.run(command2, cwd=directory_path2)
    
    # 6. save cleaned noise
    remove_noise(config.output_dir, config.outlier_dir, config.nrun)


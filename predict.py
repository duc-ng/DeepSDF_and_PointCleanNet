import argparse
import torch
import numpy as np
import os
from src.AutoDecoder import AutoDecoder
from skimage import measure
from glob import glob


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="DeepSDF and PointCleanNet")

    # for reconstruction
    parser.add_argument(
        "--input_dir", type=str, default="out/1_preprocessed", help="input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="out/2_reconstructed", help="output directory"
    )
    parser.add_argument(
        "--weights_dir", type=str, default="out/weights", help="model weights directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="device to train on")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--N", type=int, default=192, help="meshgrid size")
    parser.add_argument("--latent_size", type=int, default=128, help="latent size")
    parser.add_argument(
        "--batch_size_reconstruct", type=int, default=100000, help="batch size"
    )

    return parser.parse_args()


def save_reconstructions(model, batch_size, device, N, latent_inference, output_file):
    """
    Reconstruct the SDF values for a meshgrid and save the mesh as .obj file.
    """
    # create meshgrid
    coords = np.linspace(-1.3, 1.3, N)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    samples = np.vstack(
        (Y.flatten(), X.flatten(), Z.flatten())  # swap X,Y because of coordinate system
    ).T  # reshape to (N^3, 3)
    samples = torch.from_numpy(samples).float().to(device)

    # predict
    preds = torch.zeros(N**3)
    for i in range(0, len(samples), batch_size):
        batch = samples[i : (i + batch_size)]
        input = torch.cat((batch, latent_inference.repeat(batch.shape[0], 1)), dim=1)
        with torch.no_grad():
            pred = model(input)
            preds[i : (i + batch_size)] = pred.squeeze()
        print(f"Reconstructing mesh: {i}/{len(samples)} points", end="\r")

    # marching cubes
    preds = preds.view(N, N, N).cpu().numpy()  # reshape to (N, N, N)
    verts, faces, _, _ = measure.marching_cubes(preds, 0)

    # scale back to samples space
    grid_min = np.min(samples.cpu().numpy(), axis=0)
    grid_max = np.max(samples.cpu().numpy(), axis=0)
    verts = verts * (grid_max - grid_min) / (N - 1) + grid_min

    # save as .obj and .xyz
    with open(output_file, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Reconstructed mesh saved in: {output_file}")


if __name__ == "__main__":
    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # get filenames
    input_files = sorted(glob(os.path.join(args.input_dir, "*.npz")))
    basenames = [os.path.basename(f).split(".")[0] for f in input_files]
    weight_files = sorted(glob(os.path.join("out/weights", "*.pth")))

    # load latent and model
    latent = torch.load("out/weights/latent.pt", map_location=device).weight
    model = AutoDecoder(args.latent_size).float().train().to(device)

    # loop over names
    for i, name in enumerate(basenames):

        # create output directory
        output_dir = os.path.join(args.output_dir, f"{name}")
        os.makedirs(output_dir, exist_ok=True)

        # loop over weight checkpoints
        for weight_file in weight_files:

            # get model and latent vector
            model.load_state_dict(torch.load(weight_file, map_location=device))
            latent_inference = latent[i].unsqueeze(0)

            # get checkpoint number
            checkpoint = os.path.basename(weight_file).split(".")[0].split("_")[-1]
            output_file = os.path.join(output_dir, checkpoint + ".obj")

            # save reconstructed coordinates
            model.eval()
            save_reconstructions(
                model,
                args.batch_size_reconstruct,
                device,
                args.N,
                latent_inference,
                output_file,
            )

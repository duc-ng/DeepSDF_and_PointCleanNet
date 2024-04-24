import argparse
import torch
import numpy as np
import os
from src.AutoDecoder import AutoDecoder
from torch.utils.data import DataLoader
from src.dataset import DeepSDF_Dataset
from tqdm import tqdm
from skimage import measure
from glob import glob
import subprocess


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="DeepSDF and PointCleanNet")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="out/1_preprocessed",
        help="file to train on",
    )
    parser.add_argument(
        "--output_dir", type=str, default="out/2_reconstructed", help="output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="device to train on")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6400,
        help="input batch size for reconstruction",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument(
        "--delta", type=float, default=0.1, help="delta for clamping loss function"
    )
    parser.add_argument("--N", type=int, default=256, help="meshgrid size")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="clean the reconstruction with PointCleanNet",
    )
    parser.add_argument(
        "--clean_nrun",
        type=int,
        default=2,
        help="Number of runs with PointCleanNet noise removal",
    )
    parser.add_argument("--latent_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--latent_size", type=int, default=256, help="latent size")
    parser.add_argument("--latent_std", type=float, default=0.01, help="latent std")
    parser.add_argument("--latent_epochs", type=int, default=50, help="latent epochs")
    parser.add_argument(
        "--weights",
        type=str,
        default="out/weights/model_200.pth",
        help="load model weights",
    )
    return parser.parse_args()


def save_reconstructions(
    model, batch_size, name, device, N, output_dir, latent_inference
):
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
    grid_scale = grid_max - grid_min
    verts = verts * grid_scale / (N - 1) + grid_min

    # save as .obj and .xyz
    out_file_obj = os.path.join(output_dir, f"{name}.obj")
    out_file_xyz = os.path.join(output_dir, f"{name}.xyz")
    with open(out_file_obj, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    with open(out_file_xyz, "w") as f:
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
    print(f"Reconstructed mesh saved in: {out_file_obj} and {name}.xyz")


def estimate_latent(
    model, latent_inference, lr_latent, latent_epochs, delta, latent_std, file_index
):
    optimizer = torch.optim.Adam([latent_inference.weight], lr=lr_latent)
    criterion = torch.nn.L1Loss(reduction="mean")

    for epoch in range(latent_epochs):
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, _ in tepoch:

                # get inputs, targets and latent
                tepoch.set_description(f"Epoch {epoch+1}/{latent_epochs}")
                inputs = data[:, file_index, :3]  # coordinates
                targets = data[:, file_index, 3].unsqueeze(1)  # sdf values

                # move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                latent_vector = latent_inference.weight.to(device)

                # add latent vector to input coords
                inputs = torch.cat(
                    (inputs, latent_vector.repeat(inputs.shape[0], 1)), dim=1
                )

                # predict
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss
                targets_clamp = torch.clamp(targets, -delta, delta)
                out_clamp = torch.clamp(outputs, -delta, delta)
                regularization_term = (1 / latent_std**2) * torch.norm(
                    latent_vector, p=2
                ).pow(2)
                loss = criterion(out_clamp, targets_clamp) + regularization_term
                losses.append(loss.item())

                # backprop
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=np.mean(losses))

    return latent_inference.weight.to(device)


if __name__ == "__main__":
    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # load data
    train_dataset = DeepSDF_Dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # loop over input files
    input_files = glob(os.path.join(args.input_dir, "*.npz"))
    basenames = [os.path.basename(f).split(".")[0] for f in input_files]
    for i, name in enumerate(basenames):
        # load model
        model = AutoDecoder(args.latent_size).float().train().to(device)
        model.load_state_dict(torch.load(args.weights))

        # estimate latent vectors for inference
        latent_inference = torch.nn.Embedding(1, args.latent_size, max_norm=1.0)
        torch.nn.init.normal_(
            latent_inference.weight.data, mean=0.0, std=args.latent_std
        )
        latent_inference = estimate_latent(
            model,
            latent_inference,
            args.latent_lr,
            args.latent_epochs,
            args.delta,
            args.latent_std,
            i,
        )

        # save reconstructed coordinates
        save_reconstructions(
            model.eval(),
            args.batch_size * 10,
            name,
            device,
            args.N,
            args.output_dir,
            latent_inference,
        )

    # PointCleanNet
    if args.clean:
        command = [
            "python",
            "src/pointcleannet.py",
            "--input_dir",
            args.output_dir,
            "--outlier_dir",
            "out/3_cleaned_outliers",
            "--output_dir",
            "out/4_cleaned_noise",
            "--device",
            str(device),
            "--seed",
            str(args.seed),
            "--nrun",
            str(args.clean_nrun),
        ]
        subprocess.run(command)

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import DeepSDF_Dataset
from src.Decoder import Decoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import skimage.measure as measure
import subprocess


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="Simple Example")
    parser.add_argument(
        "--input_file",
        type=str,
        default="out/1_preprocessed/bunny.npz",
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
        help="input batch size for training and reconstruction",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument("--delta", type=float, default=0.1, help="delta for clamping")
    parser.add_argument("--N", type=int, default=32, help="meshgrid size")
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
    parser.add_argument("--load", type=str, default=None, help="load model from file instead of training")
    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, device, epochs, delta):
    """
    Train loop for the model.
    """
    print("Start training...")
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:

                # get inputs and targets
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs = data[:, 0:3]  # coordinates
                targets = data[:, 3].unsqueeze(1)  # sdf values

                # move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero gradients
                optimizer.zero_grad()

                # predict
                outputs = model(inputs)

                # Compute loss
                targets_clamp = torch.clamp(targets, -delta, delta)
                out_clamp = torch.clamp(outputs, -delta, delta)
                loss = criterion(out_clamp, targets_clamp)
                losses.append(loss.item())

                # backprop
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=np.mean(losses))

        epoch_losses.append(np.mean(losses))

    return epoch_losses


def save_plot_losses(losses):
    """
    Plot the losses.
    """
    # set output directory
    out_dir = "out/losses"
    os.makedirs(out_dir, exist_ok=True)

    # configure plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{out_dir}/loss_{len(os.listdir(out_dir))}.png")


def save_reconstructions(model, batch_size, delta, name, device, N, output_dir):
    """
    Reconstruct the SDF values for a meshgrid and save the mesh as .obj file.
    """
    # create meshgrid
    coords = np.linspace(-1.5, 1.5, N)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    samples = np.vstack(
        (Y.flatten(), X.flatten(), Z.flatten()) # swap X,Y because of coordinate system
    ).T  # reshape to (N^3, 3) 
    samples = torch.from_numpy(samples).float().to(device)

    # predict
    preds = torch.zeros(N**3)
    for i in range(0, len(samples), batch_size):
        batch = samples[i : (i + batch_size)]
        with torch.no_grad():
            pred = model(batch)
            preds[i : (i + batch_size)] = pred.squeeze()
        print(f"Reconstructing mesh: {i}/{len(samples)} vertices", end="\r")
    
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

if __name__ == "__main__":
    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # load data
    train_dataset = DeepSDF_Dataset(args.input_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # train model
    model = Decoder().float().train().to(device)
    criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    basename = os.path.basename(args.input_file).split(".")[0]
    
    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        epoch_losses = train(
            train_loader, model, criterion, optimizer, device, args.epochs, args.delta
        )
        save_plot_losses(epoch_losses)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"{basename}.pth"))    
    
    # save reconstructed coordinates as .obj and .xyz
    save_reconstructions(
        model.eval(),
        args.batch_size,
        args.delta,
        basename,
        device,
        args.N,
        args.output_dir,
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

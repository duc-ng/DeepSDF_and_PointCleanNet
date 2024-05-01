import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import DeepSDF_Dataset
from src.AutoDecoder import AutoDecoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from glob import glob


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="DeepSDF and PointCleanNet")
    parser.add_argument(
        "--weights_dir", type=str, default="out/weights", help="weights directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="device to train on")
    parser.add_argument(
        "--batch_size", type=int, default=5, help="batch size for training"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--lr_model", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--lr_latent", type=float, default=0.001, help="learning rate for latent vector"
    )
    parser.add_argument(
        "--epochs", type=int, default=2000, help="number of epochs to train"
    )
    parser.add_argument(
        "--delta", type=float, default=0.1, help="delta for clamping loss function"
    )
    parser.add_argument("--latent_size", type=int, default=128, help="latent size")
    parser.add_argument("--latent_std", type=float, default=0.01, help="latent std")
    parser.add_argument(
        "--nr_rand_samples",
        type=int,
        default=10000,
        help="Number of random subsamples from each batch",
    )

    assert parser.parse_args().batch_size <= len(
        glob("out/1_preprocessed/*.npz")
    ), "Batch size is larger than the number of files in the dataset"

    return parser.parse_args()


def train(
    train_loader,
    model,
    optimizer,
    device,
    epochs,
    delta,
    weights_dir,
    latent,
    latent_std,
):
    """
    Train loop for the model.
    """
    print("Start training...")
    epoch_losses = []
    for epoch in range(epochs):
        losses = []

        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y, indices in tepoch:

                # get inputs and targets
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")

                # move data to device
                inputs = x.to(device)
                targets = y.to(device)
                latent_vector = latent(indices).to(device)

                # add latent vector to input coords and reshape
                inputs = torch.cat(
                    (inputs, latent_vector.unsqueeze(1).repeat(1, inputs.shape[1], 1)),
                    dim=2,
                )
                inputs = inputs.view(-1, inputs.shape[2])
                targets = targets.view(-1, 1)

                # predict
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets, delta, latent_vector, latent_std)
                losses.append(loss.item())

                # backprop
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=np.mean(losses))

        epoch_losses.append(np.mean(losses))

        # save model every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, f"model_{epoch+1}.pth"),
            )

    # save latents
    torch.save(latent, os.path.join(weights_dir, "latent.pt"))

    return epoch_losses


def loss_function(y_pred, y_true, delta, latent, latent_std):
    """
    Loss function for multi-shape SDF prediction.
    """
    y_pred = torch.clamp(y_pred, -delta, delta)
    y_true = torch.clamp(y_true, -delta, delta)
    l1 = torch.mean(torch.abs(y_pred - y_true))
    l2 = latent_std**2 * torch.mean(torch.linalg.norm(latent, dim=1, ord=2))
    return l1 + l2


def save_plot_losses(losses, out_dir="out/losses"):
    """
    Plot the losses.
    """
    # set output directory
    os.makedirs(out_dir, exist_ok=True)

    # configure plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{out_dir}/loss_{len(os.listdir(out_dir))}.png")


if __name__ == "__main__":
    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.weights_dir, exist_ok=True)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # load data
    train_dataset = DeepSDF_Dataset(args.nr_rand_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # init latent vector for each sdf
    latent = torch.nn.Embedding(len(train_dataset), args.latent_size)
    torch.nn.init.normal_(latent.weight.data, mean=0.0, std=args.latent_std)

    # init model and optimizer
    model = AutoDecoder(latent_size=args.latent_size).float().train().to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": args.lr_model * args.batch_size,
            },
            {
                "params": latent.parameters(),
                "lr": args.lr_latent,
            },
        ],
    )

    # train model
    epoch_losses = train(
        train_loader,
        model,
        optimizer,
        device,
        args.epochs,
        args.delta,
        args.weights_dir,
        latent,
        args.latent_std,
    )

    # save losses
    save_plot_losses(epoch_losses)

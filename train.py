import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import DeepSDF_Dataset
from src.AutoDecoder import AutoDecoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


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
        "--batch-size", type=int, default=6400, help="batch size for training"
    )
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of epochs to train"
    )
    parser.add_argument(
        "--delta", type=float, default=0.1, help="delta for clamping loss function"
    )
    parser.add_argument("--latent_size", type=int, default=256, help="latent size")
    parser.add_argument("--latent_std", type=float, default=0.01, help="latent std")
    return parser.parse_args()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    epochs,
    delta,
    weights_dir,
    latent,
):
    """
    Train loop for the model.
    """
    print("Start training...")
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, indices in tepoch:

                # get inputs and targets
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs = data[:,:, 0:3]  # coordinates                
                targets = data[:,:, 3].unsqueeze(1)  # sdf values
                
                # move data to device
                inputs = inputs.to(device) 
                targets = targets.to(device)
                latent_vector = latent(indices).to(device) 

                # add latent vector to input coords  and reshape
                inputs = torch.cat((inputs, latent_vector.unsqueeze(1).repeat(1, inputs.shape[1], 1)), dim=2)
                inputs = inputs.view(-1, inputs.shape[2])
                targets = targets.view(-1, 1)

                # predict
                optimizer.zero_grad()
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

        # save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, f"model_{epoch+1}.pth"),
            )

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
    train_dataset = DeepSDF_Dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # init latent vector for training
    latent = torch.nn.Embedding(len(train_dataset), args.latent_size, max_norm=1.0)
    torch.nn.init.normal_(latent.weight.data, mean=0.0, std=args.latent_std)

    # init model and optimizer
    model = AutoDecoder(args.latent_size).float().train().to(device)
    criterion = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": latent.parameters()}], lr=args.lr
    )

    # train model
    epoch_losses = train(
        train_loader,
        model,
        criterion,
        optimizer,
        device,
        args.epochs,
        args.delta,
        args.weights_dir,
        latent,
    )

    # save losses
    save_plot_losses(epoch_losses)

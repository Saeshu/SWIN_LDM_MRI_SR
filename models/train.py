import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model.
    loader : DataLoader
        Training dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    device : torch.device
        Device on which the model and data are located.

    Returns
    -------
    float
        Mean reconstruction MSE over the epoch.
    """

    model.train()

    running_loss = 0.0

    pbar = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in pbar:

        # --------------------------------------------------------
        # Extract input
        # --------------------------------------------------------

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------------------
        # Forward + backward
        # --------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        x_hat = model(x)

        # Reconstruction objective
        loss = F.mse_loss(
            x_hat,
            x,
        )

        loss.backward()

        optimizer.step()

        # --------------------------------------------------------
        # Track loss
        # --------------------------------------------------------

        running_loss += (
            loss.item() * x.size(0)
        )

        pbar.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    return epoch_loss


@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    """
    Evaluate the model on a validation set.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model.
    loader : DataLoader
        Validation dataloader.
    device : torch.device
        Device on which the model and data are located.

    Returns
    -------
    float
        Mean reconstruction MSE over the validation set.
    """

    model.eval()

    running_loss = 0.0

    pbar = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in pbar:

        # --------------------------------------------------------
        # Extract input
        # --------------------------------------------------------

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        x_hat = model(x)

        # Reconstruction objective
        loss = F.mse_loss(
            x_hat,
            x,
        )

        # --------------------------------------------------------
        # Track loss
        # --------------------------------------------------------

        running_loss += (
            loss.item() * x.size(0)
        )

        pbar.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    return epoch_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=100,
    save_path="Ours_best.pt",
):
    """
    Train an autoencoder and save the best checkpoint.

    The checkpoint is selected using validation MSE.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model.
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    device : torch.device
        Training device.
    num_epochs : int, default=100
        Number of training epochs.
    save_path : str, default="Ours_best.pt"
        Path at which the best checkpoint is saved.

    Returns
    -------
    dict
        Training and validation loss history.
    """

    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):

        # ========================================================
        # Training
        # ========================================================

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # ========================================================
        # Validation
        # ========================================================

        val_loss = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        # ========================================================
        # Store history
        # ========================================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        # ========================================================
        # Save best checkpoint
        # ========================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_path,
            )

            saved = " <-- saved"

        else:
            saved = ""

        # ========================================================
        # Logging
        # ========================================================

        print(
            f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
            f"{saved}"
        )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    return history

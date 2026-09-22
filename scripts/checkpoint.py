import torch


def load_checkpoint(
    model,
    path,
    device,
    optimizer=None,
    strict=True,
):
    """
    Load a model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model into which the checkpoint is loaded.

    path : str
        Path to the checkpoint.

    device : torch.device
        Device used for loading the checkpoint.

    optimizer : torch.optim.Optimizer, optional
        Optimizer whose state should be restored.
        If None, the optimizer state is not loaded.

    strict : bool, default=True
        Whether to require an exact match between the
        checkpoint and model state dictionaries.

    Returns
    -------
    int
        Epoch from which training should resume.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # ============================================================
    # Model
    # ============================================================

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    # ============================================================
    # Optimizer
    # ============================================================

    if optimizer is not None:

        if "optimizer_state_dict" in checkpoint:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            print(
                "Optimizer state loaded."
            )

        else:

            print(
                "No optimizer state found "
                "in checkpoint."
            )

    # ============================================================
    # Checkpoint information
    # ============================================================

    epoch = checkpoint.get(
        "epoch",
        0,
    )

    train_loss = checkpoint.get(
        "train_loss",
        None,
    )

    val_loss = checkpoint.get(
        "val_loss",
        None,
    )

    # ============================================================
    # Logging
    # ============================================================

    print("\n" + "=" * 50)
    print("CHECKPOINT LOADING")
    print("=" * 50)

    print(
        f"Checkpoint: {path}"
    )

    print(
        f"Epoch:      {epoch}"
    )

    if train_loss is not None:
        print(
            f"Train loss: {train_loss:.6f}"
        )

    if val_loss is not None:
        print(
            f"Val loss:   {val_loss:.6f}"
        )

    print(
        "Model state: loaded successfully"
    )

    print("=" * 50)

    return epoch + 1

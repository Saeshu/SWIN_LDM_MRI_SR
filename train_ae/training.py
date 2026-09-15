import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0

    pbar = tqdm(loader, desc="Training")

    for batch in pbar:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        x_hat = model(x)

        loss = F.mse_loss(
            x_hat,
            x,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss


@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    running_loss = 0.0

    pbar = tqdm(loader, desc="Validation")

    for batch in pbar:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(
            device,
            non_blocking=True,
        )

        x_hat = model(x)

        loss = F.mse_loss(
            x_hat,
            x,
        )

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(loader.dataset)

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
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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

import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device='cpu',
    epochs=10,
    checkpoint_dir='checkpoints'
):
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        model.train()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=120
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / running_total

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device
        )

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir + "/best"
            )

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Acc={train_accuracy:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Acc={val_accuracy:.2f}%"
        )

    return model


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device='cpu',
    epochs=10
):
    model.to(device)

    for epoch in range(epochs):

        running_loss = 0.0

        model.train()

        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=120
        )

        for inputs, _ in progress_bar:

            inputs = inputs.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar = model(inputs)

            loss = criterion(
                reconstruction,
                inputs,
                mu,
                logvar
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = running_loss / len(data_loader)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Average Loss: {avg_loss:.4f}"
        )

    print("Finished VAE Training")

    return model
import math
import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10):
    model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, 2).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()

    cols = min(6, num_samples)
    rows = math.ceil(num_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')

    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
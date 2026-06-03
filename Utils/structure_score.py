import torch.nn.functional as F

def structure_score(x):
    # x: [D, H, W]
    img = x[x.shape[0] // 2]
    img = img - img.mean()
    img = img / (img.std() + 1e-8)

    shifts = [2, 4, 8, 16, 32]
    scores = []

    for s in shifts:
        shifted = torch.roll(img, shifts=s, dims=1)
        sim = F.cosine_similarity(
            img.flatten(),
            shifted.flatten(),
            dim=0
        )
        scores.append(sim.item())

    return sum(scores) / len(scores)

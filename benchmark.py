import statistics, time
import torch
import torch.nn.functional as F

from models.simple_gan import SimpleGenerator, SimpleDiscriminator
from models.dit_gan import DiTGenerator, Discriminator as DiTDiscriminator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(0)


def params(m):
    return sum(p.numel() for p in m.parameters())


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def make_latent(g, latent_dim, batch):
    z = torch.randn(batch, latent_dim, device=device)
    try:
        with torch.no_grad():
            g(z)
        return z
    except Exception:
        return torch.randn(batch, latent_dim, 1, 1, device=device)


def bench(name, G, D, latent_dim, batch, runs=5):
    G, D = G.to(device), D.to(device)
    print(f"\n=== {name} ===")
    print(f"Generator params: {params(G):,}   Discriminator params: {params(D):,}")

    # Generation speed
    G.eval()
    z = make_latent(G, latent_dim, batch)
    with torch.no_grad():
        out = G(z)
        print(f"Output shape: {tuple(out.shape)}")
        G(z); sync()
        times = []
        for _ in range(runs):
            t = time.perf_counter(); G(z); sync()
            times.append(time.perf_counter() - t)
    med = statistics.median(times)
    print(f"Generation: {batch / med:.1f} images/sec (batch {batch}, median of {runs})")

    # Training-step speed on synthetic data (hinge loss, so it works with any output)
    G.train(); D.train()
    optG = torch.optim.Adam(G.parameters(), 2e-4)
    optD = torch.optim.Adam(D.parameters(), 2e-4)
    real = torch.randn_like(out)
    times = []
    for i in range(runs + 1):
        t = time.perf_counter()
        fake = G(z)
        d_loss = F.relu(1 - D(real)).mean() + F.relu(1 + D(fake.detach())).mean()
        optD.zero_grad(); d_loss.backward(); optD.step()
        g_loss = -D(fake).mean()
        optG.zero_grad(); g_loss.backward(); optG.step()
        sync()
        if i > 0:  # skip warmup
            times.append(time.perf_counter() - t)
    step = statistics.median(times)
    epoch = step * (3000 / batch)
    print(f"Training step: {step:.3f}s (batch {batch}); about {epoch / 60:.1f} min per epoch on 3,000 images")


bench("SimpleGAN (128px)", SimpleGenerator(), SimpleDiscriminator(), 100, batch=16)
bench("DiT-GAN (256px)", DiTGenerator(), DiTDiscriminator(), 512, batch=4)
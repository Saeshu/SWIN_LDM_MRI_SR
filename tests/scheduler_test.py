import matplotlib.pyplot as plt
from Diffusion.LinearNoise import NoiseScheduler

noise_scheduler = NoiseScheduler(
                    timesteps = 50,
                    prediction_type = 'v'
                  )


print("Betas")
print(noise_scheduler.betas)

print("Alphas")
print(noise_scheduler.alphas)

print("Alpha bars")
print(noise_scheduler.alpha_bars)

assert torch.all(noise_scheduler.betas > 0)
assert torch.all(noise_scheduler.alphas < 1)
assert torch.all(noise_scheduler.alpha_bars[:-1] > noise_scheduler.alpha_bars[1:])
device = "cuda"

x0 = torch.randn(
    512,
    1,
    16,
    16,
    16,
    device=device,
)

noise = torch.randn_like(x0)

for t in [0,10,20,30,40,49]:

    t_batch = torch.full((x0.shape[0],), t, device=device)

    xt = noise_scheduler.add_noise(
        x0,
        t_batch,
        noise,
    )

    print(
        "test 1: ",
        f"t={t:2d}",
        "mean =", xt.mean().item(),
        "std =", xt.std().item(),
    )
for t in range(noise_scheduler.num_timesteps):

    t_batch = torch.full((x0.shape[0],), t, device=device)

    xt = noise_scheduler.add_noise(
        x0,
        t_batch,
    )

    print(
        "test 2: ",
        t,
        xt.var().item(),
    )
  
v = noise_scheduler.get_velocity(
    x0,
    noise,
    t_batch,
)

x0_hat = noise_scheduler.predict_x0(
    xt,
    v,
    t_batch,
)
print("test 3:")
print(
    ((x0_hat-x0)**2).mean()
)

v = noise_scheduler.get_velocity(
    x0,
    noise,
    t_batch,
)

eps_hat = noise_scheduler.predict_eps(
    xt,
    v,
    t_batch,
)

print(
    ((eps_hat-noise)**2).mean()
)

v = noise_scheduler.get_velocity(
    x0,
    noise,
    t_batch,
)

eps = noise_scheduler.predict_eps(
    xt,
    v,
    t_batch,
)

x0_hat = noise_scheduler.predict_x0(
    xt,
    v,
    t_batch,
)

v2 = noise_scheduler.get_velocity(
    x0_hat,
    eps,
    t_batch,
)

print(
    ((v-v2)**2).mean()
)

model_output = noise_scheduler.get_velocity(
    x0,
    noise,
    t_batch,
)

xt_prev = noise_scheduler.step(
    xt,
    model_output,
    t_batch,
)

t_prev = torch.clamp(t_batch-1,min=0)

xt_true = noise_scheduler.add_noise(
    x0,
    t_prev,
    noise,
)
print("test 4: ") 
print(
    ((xt_prev-xt_true)**2).mean()
)

T = noise_scheduler.num_timesteps

samples = torch.randint(
    0,
    T,
    (100000,),
)
print("test 5")
plt.hist(
    samples.numpy(),
    bins=T,
)

plt.show()
print("test 6")
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.plot(noise_scheduler.betas.cpu())
plt.title("Beta")

plt.subplot(122)
plt.plot(noise_scheduler.alpha_bars.cpu())
plt.title("Alpha Bar")

plt.show()

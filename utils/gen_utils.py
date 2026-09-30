from models import DDPM

import torch

import torchutils

@torch.no_grad()
def gen(ddpm: DDPM, z: torch.Tensor, dtype: str, device: torch.device) -> torch.Tensor:
    dtype = torchutils.get_torch_dtype(dtype)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)

    eta = 0.0
    ddpm.eval()
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=use_amp
    ):
        batch_size = z.shape[0]
        z = torch.randn(
            (batch_size, *ddpm.img_size),
            device=device,
            dtype=dtype,
        )
        times = torch.linspace(
            ddpm.num_timesteps - 1,
            0,
            ddpm.num_sampling_timesteps,
            dtype=torch.long,
            device=device,
        )
        for i in range(len(times) - 1):
            t = times[i].item()
            t_prev = times[i + 1].item()
            timesteps = torch.full(
                (z.shape[0],),
                t,
                dtype=torch.long,
                device=device,
            )
            prev_timesteps = torch.full(
                (z.shape[0],),
                t_prev,
                dtype=torch.long,
                device=device,
            )
            _, x0 = ddpm(z, timesteps)
            z = z.to(dtype)
            x0 = x0.to(dtype)
            alpha_t = (
                ddpm.alphas_cumprod.gather(
                    -1,
                    timesteps
                )
                .reshape(
                    timesteps.shape[0],
                    *((1,) * (len(z.shape)-1))
                )
            )
            alpha_prev = (
                ddpm.alphas_cumprod.gather(
                    -1,
                    prev_timesteps
                )
                .reshape(
                    prev_timesteps.shape[0],
                    *((1,) * (len(z.shape)-1))
                )
            )
            eps = (
                z - torch.sqrt(alpha_t) * x0
            ) / torch.sqrt(1.0 - alpha_t)
            sigma = (
                eta
                * torch.sqrt(
                    (1.0 - alpha_prev)
                    / (1.0 - alpha_t)
                    * (1.0 - alpha_t / alpha_prev)
                )
            )
            direction = torch.sqrt(
                1.0 - alpha_prev - sigma ** 2
            ) * eps
            noise = (
                sigma * torch.randn_like(z)
                if eta > 0 and i < len(times)-2
                else 0
            )
            z = (
                torch.sqrt(alpha_prev) * x0
                + direction
                + noise
            )
    ddpm.train()
    images = z / 2 + 0.5
    images = images * 255
    return images
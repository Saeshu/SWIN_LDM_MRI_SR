import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
def compute_losses(
    ae,
    x0_pred,
    z_res,
    z_hr,
    z_lr,
    v_pred,
    v_target,
    hr,
    perc_net,
    alpha_bar,
    noise,
):
  proj_layer = None
  # -----------------------------
  # Training setup
  # -----------------------------
  MIN_EMA_STEPS = 100
  accum_steps = 4
  num_epochs = 16

  epoch_bar = tqdm(range(num_epochs), desc="Epochs")


  for epoch in epoch_bar:
      ae.eval()
      unet.train()

      # 🔥 Running stats (FIXED logging)
      loss_sum = 0.0
      mse_sum = 0.0
      std_sum = 0.0
      count = 0

      optimizer.zero_grad(set_to_none=True)
      global_step = 0

      indices = torch.randperm(len(train_ds))
      subset_size = int(0.5 * len(train_ds))
      subset_indices = indices[:subset_size]

      for idx in tqdm(indices, desc="batches", leave=False):

          # -----------------------------
          # Load data
          # -----------------------------
          hr, lr = train_ds[idx]
          hr = hr.unsqueeze(0).to(device)
          lr = lr.unsqueeze(0).to(device)
          with torch.no_grad():
              # -----------------------------
              # Encode
              # -----------------------------
              z_hr_out = ae.encode(hr)
              z_lr_out = ae.encode(lr)

              z_lr_out = z_lr_out[0] if isinstance(z_lr_out, (list, tuple)) else z_lr_out
              z_hr_out = z_hr_out[0] if isinstance(z_hr_out, (list, tuple)) else z_hr_out

              # -----------------------------
              # 🔥 Latent projection (replace mean)
              # -----------------------------
              target_shape = latent_shape[2:]

              z_hr = F.interpolate(z_hr_out, size=target_shape, mode="trilinear", align_corners=False)
              z_lr = F.interpolate(z_lr_out, size=target_shape, mode="trilinear", align_corners=False)

              # 🔥 apply adapter (residual form)
              z_hr, delta_hr = adapter(z_hr)
              z_lr, delta_lr = adapter(z_lr)
              # -----------------------------
              # Residual
              # -----------------------------
              z_res = z_hr - z_lr
              z_final_gt = z_res + z_lr
              # -----------------------------
              # Residual
              # -----------------------------
              z_res = z_hr - z_lr
              z_final_gt = z_res + z_lr
          # ---- Encode ----
          with torch.no_grad():
              z = z_hr
              cond = z_lr
              #print(z_res.shape)
              #print("hr small shape",z_hr_small.shape)

          z = z.detach()
          cond = cond.detach()
          z_res = z_res.detach()
          #z_hr_small1 = z_hr_small1.detach()



          # ---- timestep sampling (BIASED toward low-noise) ----
          T = noise_sched.num_timesteps
          u = torch.rand(z.shape[0], device=z.device)
          t = ((u**2) * int(0.4 * T)).long()
          '''
          if torch.rand(1) < 0.9:
              # 🔥 strong low-t focus (refinement)
              t = ((u**2) * int(0.3 * T)).long()
          else:
              # 🌍 global coverage (stability)
              t = (u * 0.6*T).long()
          '''
          t_norm = t.float() / T

          # ---- forward diffusion ----
          noise = torch.randn_like(z_hr)
          z_noisy = noise_sched.add_noise(z_hr, t, noise)
          #z_noisy = z_noisy[:, :4]
          #print(z_noisy.shape)
          #print(cond.shape)
          # ---- alpha_bar ----
          alpha_bar_t = noise_sched.alpha_bars[t].to(z.device)
          alpha_bar_t = torch.clamp(alpha_bar_t, 1e-5, 1 - 1e-5)
          alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1, 1)


          # ---- v-target ----
          v = (
          torch.sqrt(alpha_bar_t) * noise
          - torch.sqrt(1.0 - alpha_bar_t) * z
          )


          # ---- conditioning dropout ----
          cond_input = None if torch.rand(1).item() < 0.2 else cond


          # ---- fixed alpha ----
          alpha = 0.4


          # -----------------------------
          # Forward (AMP)
          # -----------------------------
          with autocast("cuda"):
              cond_add = cond.detach()   # 🔥 break graph here
              v_pred = unet(z_noisy, t, cond=z_lr, alpha=alpha)
              v_pred = torch.clamp(v_pred, -4.0, 4.0)

              # -----------------------------
              # Core losses (per-pixel for weighting)
              # -----------------------------
              # -----------------------------
              # x0 prediction
              # -----------------------------
              x0_pred = (
              z_noisy - torch.sqrt(1 - alpha_bar_t) * v_pred
              ) / torch.sqrt(alpha_bar_t)

              #x0_pred = x0_pred / (x0_pred.std().detach() + 1e-6)
              #x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
              x0_pred = torch.tanh(x0_pred)
              scale = (z_res.std(dim=(2,3,4), keepdim=True) /
              (x0_pred.std(dim=(2,3,4), keepdim=True) + 1e-6)).detach()

              x0_pred = x0_pred * scale
              x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
              # -----------------------------
              # Core losses
              # -----------------------------
              mse_map = (v_pred - v) ** 2
              mse = mse_map.mean()
              # 1. diffusion loss (primary)
              mse_loss = F.mse_loss(v_pred, v)

              # 2. gradient / structure (lightweight)
              #g_loss = grad_loss_light(v_pred, v)

              # 3. reconstruction (anchors latent)
              #recon_loss = F.mse_loss(x0_pred, z.detach())

              # -----------------------------
              # Build perc inputs (MATCH training)
              # -----------------------------

              # ----- PRED -----
              x_base = x0_pred.mean(dim=1, keepdim=True)   # [B,1,...]
              z_base = z_res.mean(dim=1, keepdim=True)


              x_struct = F.avg_pool3d(x_base, 2)

              x_blur = F.avg_pool3d(x_struct, 2)
              x_up = F.interpolate(x_blur, size=x_struct.shape[2:], mode="trilinear", align_corners=False)

              x_detail = x_struct - x_up

              x_input = torch.cat([x_struct, x_detail], dim=1)


              # ----- GT -----


              z_struct = F.avg_pool3d(z_base, 2)

              z_blur = F.avg_pool3d(z_struct, 2)
              z_up = F.interpolate(z_blur, size=z_struct.shape[2:], mode="trilinear", align_corners=False)

              z_detail = z_struct - z_up

              z_input = torch.cat([z_struct, z_detail], dim=1)


              # -----------------------------
              # Forward through perc_net
              # -----------------------------
              f_pred, _ = perc_net(x_input)

              with torch.no_grad():
                  f_gt, _ = perc_net(z_input)


              # -----------------------------
              # Channel-aware perceptual loss
              # -----------------------------


              # perceptual
              perc_struct = torch.mean(torch.abs(f_pred[:, :4] - f_gt[:, :4]))
              perc_detail = torch.mean(torch.abs(f_pred[:, 4:] - f_gt[:, 4:]))
              perc_loss = 0.02 * perc_struct + 0.05 * perc_detail

              # high-frequency
              '''hf_pred = high_freq(x_base)
              hf_gt   = high_freq(z_base.detach())

              mask = torch.sigmoid(10 * (hf_gt.abs() - 0.05))
              hf_loss = torch.mean(mask * torch.abs(hf_pred - hf_gt))
              hf_loss = torch.sum(mask * torch.abs(hf_pred - hf_gt)) / (mask.sum() + 1e-6)
              '''
              # patch
              patch_loss = random_patch_loss(x_base, z_base.detach(), num_patches=8)
              edge_loss = F.l1_loss(
                  edge_map(x_base),
                  edge_map(z_base.detach())
              )
              # direction
              '''
              dir_loss = gradient_direction_loss(
                  F.avg_pool3d(x_base, 2),
                  F.avg_pool3d(z_base.detach(), 2)
              )
              '''
              #print("x0 pred", x0_pred.shape)
              #print("z_lr", z_lr.shape)
              z_final = cond + 0.12 * x0_pred
              #x0_pred_small = x0_pred.mean(dim=1, keepdim=True)
              #z_res_small   = z_res.mean(dim=1, keepdim=True)
              #z_final_small = z_final.mean(dim=1, keepdim=True)
              recon_loss = F.l1_loss(z_final, z_hr)
              scale_loss = torch.abs(x0_pred.std() - z_res.std())
              res_loss = F.l1_loss(x0_pred, z_res)


              perc_detail_loss = torch.mean(
                  (f_pred[:, 4:] - f_gt[:, 4:])**2
              )


              cos_loss = 1 - F.cosine_similarity(
                  f_pred[:, 4:].flatten(1),
                  f_gt[:, 4:].flatten(1),
                  dim=1
              ).mean()

              x0_loss = F.l1_loss(
              x0_pred + z_lr,
              z_hr,
              reduction="none"
          )
              '''
              with torch.no_grad():
                  x_pred_img = ae.decode(x0_pred)
              #x_pred_img = ae.decode(x0_pred)
              x_pred_img = F.interpolate(
              x_pred_img,
              size=hr.shape[2:],   # match GT
              mode="trilinear",
              align_corners=False
          )
              '''



              def blur(x):
                  return F.avg_pool3d(x, kernel_size=(1,5,5), stride=1, padding=(0,2,2))

              lf_loss = F.l1_loss(blur(x0_pred), blur(z_res.detach()))

              #fft_l = fft_loss_highfreq(x_pred_img, hr)
              def lap(x):
                  return x - F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)

              hf_pred = lap(x0_pred)
              hf_gt   = lap(z_res.detach())

              mask = torch.sigmoid(10 * (hf_gt.abs() - 0.02))  # 🔥 threshold real edges

              hf_loss = (mask * (hf_pred - hf_gt).abs()).sum() / (mask.sum() + 1e-6)

              def reduce_loss(x):
                  return x.mean(dim=tuple(range(1, x.ndim))) if x.ndim > 1 else x

              res_loss = reduce_loss(res_loss)
              perc_loss = reduce_loss(perc_loss)
              patch_loss = reduce_loss(patch_loss)
              #edge_loss = reduce_loss(edge_loss)
              #std_loss = reduce_loss(std_loss)
              #sharp_loss = reduce_loss(sharp_loss)
              #feat_std_loss = reduce_loss(feat_std_loss)
              perc_detail_loss = reduce_loss(perc_detail_loss)
              cos_loss = reduce_loss(cos_loss)
              x0_loss = reduce_loss(x0_loss)
              hf_loss = reduce_loss(hf_loss)
              recon_loss = reduce_loss(recon_loss)



              # -------------------
              # total loss
              # -------------------
              #loss = 0.3 * mse_loss
              #loss += 0.03 * g_loss
              total_loss = (
                  0.7 * mse_loss
                  + 0.25 * perc_loss          # ↑ slightly stronger perceptual
                  + 0.06 * hf_loss          # 🔥 increase HF pressure
                  + 0.25 * patch_loss       # ↓ slightly (avoid oversmoothing)
                  + 0.3  * recon_loss       # ↑ better latent alignment
                  + 0.6  * res_loss         # keep strong but not dominant
                  + 0.05 * cos_loss
                  + 0.6  * x0_loss

                  #+ 0.005 * lf_loss
                  #+ 0.01 * fft_l
              )
              #x0_pred_small = x0_pred.mean(dim=1, keepdim=True)
              #if epoch < 3:
              #loss += 1.0 * F.l1_loss(x0_pred_small, z_res.detach())
              #else:
              #   loss += 0.5 * F.l1_loss(x0_pred_small, z_res.detach())
              if idx % 4 == 0:
                  with torch.no_grad():
                      pred_img = ae.decode(x0_pred[:1])   # keep it lightweight
                      gt_img   = hr[:1]

                      img_loss = F.l1_loss(pred_img, gt_img)
                      total_loss += 0.01 * img_loss
              w = (1 - t_norm) ** 3
              w = w / (w.mean() + 1e-6)

              loss = (w * total_loss).mean()
              loss_unscaled = loss.item()

              loss_sum += loss_unscaled
              mse_sum += mse.item()
              std_sum += v_pred.std().item()
          count += 1

          # -----------------------------
          # Debug logging (SAFE)
          # -----------------------------
          if idx % 200 == 0:
              tqdm.write(
                  f"[epoch {epoch} | idx {idx}] "
                  f"loss={loss_unscaled:.4f} | "
                  f"mse={mse.item():.3f} | "
                  f"pred_std={v_pred.std().item():.3f} |"
                  f"z_res_std={z_res.std().item():.3f} |"
                  f"x0_pred std={x0_pred.std().item()} |"
                  f"z_res std={z_res.std().item()}|"
              )
              with torch.no_grad():
                  #with torch.no_grad():
                  slice_idx = x0_pred.shape[2] // 2
                  z_slice = x0_pred[:1, :, slice_idx:slice_idx+1].detach().cpu()

                  ae_cpu = ae.cpu()
                  img_slice = ae_cpu.decode(z_slice)
                  ae.to(device)
                  img_slice = F.interpolate(
                      img_slice,
                      size=(1, 256, 256),
                      mode="trilinear",
                      align_corners=False
                  )


                  hr_slice = hr[:1, :, slice_idx:slice_idx+1].cpu()
                  plot_error_heatmap(img_slice, hr_slice)
                  plot_hf_error(img_slice, hr_slice)
                  mag_pred, mag_gt = fft_analysis(x0_pred.detach(), z_hr.detach())

                  # FFT image
                  plot_fft(mag_pred, mag_gt)

                  # radial profile
                  r_pred = radial_profile(mag_pred)
                  r_gt   = radial_profile(mag_gt)

                  import matplotlib.pyplot as plt
                  plt.plot(r_pred, label="pred")
                  plt.plot(r_gt, label="gt")
                  plt.legend()
                  plt.title("Frequency profile")
                  plt.show()

          # 🔥 Lighter diff check (less VRAM)




          # -----------------------------
          # Backprop
          # -----------------------------
          loss = loss / accum_steps
          scaler.scale(loss).backward()
          global_step += 1

          if global_step % accum_steps == 0:
              scaler.unscale_(optimizer)
              torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad(set_to_none=True)

              ema.update(unet)

              # 🔥 SAFE CLEANUP
              del z, z_noisy, v, v_pred, noise
    return {
            "loss": total_loss,
            "mse": mse_loss,
            "recon": recon_loss,
            "res": res_loss,
            "perc": perc_loss,
        }

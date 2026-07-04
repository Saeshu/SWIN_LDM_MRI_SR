import torch
import torch.nn.functional as F


def train_step(
    self,
    hr: torch.Tensor,
    lr: torch.Tensor,
    debug: bool = True,
):
    """
    Perform one diffusion optimization step.

    Parameters
    ----------
    hr : [B,1,D,H,W]

    lr : [B,1,D,H,W]

    Returns
    -------
    dict
    """

    ########################################################
    # Device
    ########################################################

    hr = hr.to(self.device)
    lr = lr.to(self.device)

    ########################################################
    # Encode
    ########################################################

    with torch.no_grad():

        encoded = self._encode(hr, lr)

    ########################################################
    # Forward diffusion
    ########################################################

    outputs = self._forward(encoded)

    ########################################################
    # Loss
    ########################################################

    losses = self._compute_loss(outputs)

    ########################################################
    # Optimizer
    ########################################################

    # self._optimizer_step(losses["total"])

    ########################################################
    # Return
    ########################################################

    result = {
        
        "loss": losses["total"],

        "losses": losses,
        
        "z_hr": outputs["z_hr"],

        "z_lr": outputs["z_lr"],

        "z_res": outputs["z_res"],

        "w_e2": outputs["w_e2"],

        "t": outputs["t"],

        "noise": outputs["noise"],

        "alpha_bar": outputs["alpha_bar"],

        "z_noisy": outputs["z_noisy"],

        "v_target": outputs["v_target"],

        "v_pred": outputs["v_pred"],

        "x0_pred": outputs["x0_pred"],

    }

    if debug:

        result.update({

        "z_hr": outputs["z_hr"],

        "z_lr": outputs["z_lr"],

        "z_res": outputs["z_res"],

        "w_e2": outputs["w_e2"],

        "t": outputs["t"],

        "noise": outputs["noise"],

        "alpha_bar": outputs["alpha_bar"],

        "z_noisy": outputs["z_noisy"],

        "v_target": outputs["v_target"],

        "v_pred": outputs["v_pred"],

        "x0_pred": outputs["x0_pred"],


        })

    return result

def _encode(self, hr, lr):

    z_hr, w_e2 = self.ae.encode(hr)
    z_lr, _ = self.ae.encode(lr)
    z_lr = F.interpolate(
    z_lr,
    size=z_hr.shape[2:],
    mode="trilinear",
    align_corners=False,
)
    z_res = z_hr - z_lr
    encoded = {
        "z_hr": z_hr,

        "z_lr": z_lr,

        "z_res": z_res,

        "w_e2": w_e2,}
    
    return encoded 

    

def _forward(self, encoded):

    z_hr = encoded["z_hr"]
    z_lr = encoded["z_lr"]
    z_res = encoded["z_res"]
    w_e2 = encoded["w_e2"]

    ####################################################
    # timestep
    ####################################################

    t = self.sample_timesteps(
        z_hr.shape[0]
    )

    ####################################################
    # noise
    ####################################################

    noise = torch.randn_like(z_res)

    z_noisy = self.scheduler.add_noise(
        z_res,
        t,
        noise,
    )

    ####################################################
    # target
    ####################################################

    alpha_bar = self.scheduler.alpha_bars[t].view(
        -1,1,1,1,1
    )

    v_target = (
        torch.sqrt(alpha_bar) * noise
        -
        torch.sqrt(1-alpha_bar) * z_res
    )

    ####################################################
    # prediction
    ####################################################

    with torch.cuda.amp.autocast():

        x0_pred = self.unet(

            z=z_noisy,

            t=t,

            cond=z_lr,

            w_e2=w_e2,

            alpha=self.cfg_scale,

        )
        alpha_bar = outputs["alpha_bar"]   # shape [B,1,1,1,1]

        v_from_x0 = (
            torch.sqrt(alpha_bar) * noise
            -
            torch.sqrt(1.0 - alpha_bar) * x0_pred
        )
        # print("w_e2:", w_e2 is None)
        
        
        # x0_pred = predict_x0(

        #     z_noisy,

        #     v_pred,

        #     alpha_bar,

        # )
    # print("v_pred :", v_pred.std())
    # print("v_target :", v_target.std())
    # print("x0_pred :", x0_pred.std())
    # print("z_res :", z_res.std())
    # print("z_hr :", z_hr.std())
    # print("z_lr :", z_lr.std())
    
    return {

        "z_hr": z_hr,

        "z_lr": z_lr,

        "z_res": z_res,

        "w_e2": w_e2,

        "t": t,

        "noise": noise,

        "alpha_bar": alpha_bar,

        "z_noisy": z_noisy,

        "v_target": v_target,

        "v_pred": v_from_x0,

        "x0_pred": x0_pred,

    }

def _compute_loss(self, outputs):
    
    return compute_losses(
        
        v_pred=outputs["v_pred"],

        v_target=outputs["v_target"],

        x0_pred=outputs["x0_pred"],

        residual_gt=outputs["z_res"],

        z_hr = outputs["z_hr"],

    )

# def _optimizer_step(self, loss):

#     self.optimizer.zero_grad(set_to_none=True)

#     self.scaler.scale(loss).backward()

#     self.scaler.unscale_(self.optimizer)

#     torch.nn.utils.clip_grad_norm_(

#         self.unet.parameters(),

#         1.0,

#     )

#     self.scaler.step(self.optimizer)

#     self.scaler.update()

#     self.ema.update(self.unet)

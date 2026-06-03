import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0          # 🔑 NEW: count EMA updates

        # independent EMA model
        self.ema_model = copy.deepcopy(model).eval()

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights from the given model.
        Call this AFTER optimizer.step().
        """
        self.num_updates += 1         # 🔑 increment counter

        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():
            esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    def state_dict(self):
        """
        Save EMA state (weights + update count).
        """
        return {
            "ema_model": self.ema_model.state_dict(),
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        """
        Restore EMA state.
        """
        self.ema_model.load_state_dict(state_dict["ema_model"], strict=True)

        self.num_updates = state_dict.get("num_updates", 0)
        self.decay = state_dict.get("decay", self.decay)

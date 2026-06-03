def load(path):
    # =========================
    # Optimizer + scaler (MUST be before loading)
    # =========================
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-5,
        weight_decay=1e-4
    )
    
    from torch.amp import GradScaler
    scaler = GradScaler()
    
    start_epoch = 0
    loss_history = []
    CKPT_PATH = path
    # =========================
    # Load checkpoint
    # =========================
    ckpt = torch.load(CKPT_PATH, map_location=device)
    
    unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
    ema.load_state_dict(ckpt["ema_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
    start_epoch = ckpt["epoch"]
    loss_history = ckpt.get("loss_history", [])
    
    tqdm.write(f"✅ Resumed from epoch {start_epoch}")

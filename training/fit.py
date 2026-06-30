def fit(
    self,
    train_loader,
    # val_loader,
    epochs,
    start_epoch=0,
):
    """
    Main training loop.

    Parameters
    ----------
    train_loader
    val_loader
    epochs
    start_epoch
    """

    ##########################################################
    # Best validation loss
    ##########################################################

    best_metric = float("inf")

    ##########################################################
    # Epoch loop
    ##########################################################

    for epoch in range(start_epoch, epochs):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}/{epochs}")
        print("=" * 70)

        ######################################################
        # Train
        ######################################################

        train_stats = self.train_epoch(train_loader)

        ######################################################
        # Validation
        ######################################################

        # val_stats = self.validate_epoch(val_loader)

        ######################################################
        # Callbacks
        ######################################################

        # use first validation sample
        hr, lr = next(iter(train_loader))

        self.run_callbacks(
            epoch=epoch,
            hr=hr,
            lr=lr,
        )

        ######################################################
        # Logger summary
        ######################################################

        print("\nTraining")

        for key, value in train_stats.items():
            print(f"{key:25s}: {value:.5f}")

        # print("\nValidation")

        # for key, value in val_stats.items():
        #     print(f"{key:25s}: {value:.5f}")

        ######################################################
        # Save last checkpoint
        ######################################################

        # save_last(
        #     model=self.unet,
        #     optimizer=self.optimizer,
        #     scheduler=self.scheduler,
        #     scaler=self.scaler,
        #     ema=self.ema,
        #     epoch=epoch,
        #     save_dir=self.save_dir,
        # )

        ######################################################
        # Save best checkpoint
        ######################################################

        # metric = val_stats.get(
        #     "l1",
        #     val_stats.get("loss", 0.0),
        # )

        # best_metric = save_best(
        #     model=self.unet,
        #     metric=metric,
        #     best_metric=best_metric,
        #     optimizer=self.optimizer,
        #     scheduler=self.scheduler,
        #     scaler=self.scaler,
        #     ema=self.ema,
        #     epoch=epoch,
        #     save_dir=self.save_dir,
        # )

    ##########################################################
    # Finished
    ##########################################################

    print("\nTraining complete.")

    # return self.logger.history

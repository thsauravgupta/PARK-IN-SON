def run_loss_weight_trial(
    cls_weight,
    hsic_weight,
    seed=42,
):
    ...
    return {
        "cls_weight": cls_weight,
        "hsic_weight": hsic_weight,
        "best_round": best_round,
        "val_ccc": val_ccc,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
    }
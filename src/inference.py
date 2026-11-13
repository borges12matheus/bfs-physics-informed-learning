import numpy as np
import pandas as pd
import torch


@torch.inference_mode()
def predict_fields(
    model,
    df: pd.DataFrame,
    feat_cols: list[str],
    xscaler,
    yscaler,
    batch_size: int = 4096,
) -> pd.DataFrame:

    model.eval()
    device = next(model.parameters()).device

    x_mu, x_sd = xscaler
    y_mu, y_sd = yscaler

    x_mu = np.asarray(x_mu, dtype=np.float32)
    x_sd = np.asarray(x_sd, dtype=np.float32)
    y_mu = np.asarray(y_mu, dtype=np.float32)
    y_sd = np.asarray(y_sd, dtype=np.float32)

    x_sd = np.where(np.abs(x_sd) > 1e-12, x_sd, 1.0)

    X = df[feat_cols].to_numpy(dtype=np.float32)
    Xn = (X - x_mu) / x_sd

    predictions = []

    for i0 in range(0, len(df), batch_size):
        xb = torch.as_tensor(
            Xn[i0:i0 + batch_size],
            dtype=torch.float32,
            device=device,
        )

        pred_n = model(xb)[:, :3]

        predictions.append(
            pred_n.cpu().numpy()
        )

    pred_n = np.concatenate(predictions, axis=0)
    pred = pred_n * y_sd[:3] + y_mu[:3]

    result = df.copy()

    result["dUx_pred"] = pred[:, 0]
    result["dUy_pred"] = pred[:, 1]
    result["dp_pred"] = pred[:, 2]

    result["Ux_corr"] = result["Ux"] + result["dUx_pred"]
    result["Uy_corr"] = result["Uy"] + result["dUy_pred"]
    result["p_corr"] = result["p"] + result["dp_pred"]

    # Campos de referência, quando disponíveis
    if {"dUx", "dUy", "dp"}.issubset(result.columns):
        result["Ux_f"] = result["Ux"] + result["dUx"]
        result["Uy_f"] = result["Uy"] + result["dUy"]
        result["p_f"] = result["p"] + result["dp"]

    return result
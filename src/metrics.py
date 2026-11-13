# Inicializa as bibliotecas necessárias
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inference import predict_fields

# ----------------------------
# Métricas
# ----------------------------
def mae_vec(u1, v1, u2, v2):
        return float(np.mean(np.sqrt((u1 - u2)**2 + (v1 - v2)**2)))
    
def mae_vec_pressure(p1, p2):
    return float(np.mean(np.abs(p1 - p2)))

def rmse_uv(u1, v1, u2, v2):
    return float(np.sqrt(np.mean((u1-u2)**2 + (v1-v2)**2)))

def rmse_p(p1, p2):
    return float(np.sqrt(np.mean((p1-p2)**2)))

def l2_relative_uv(u_pred, v_pred, u_ref, v_ref):
    num = np.sqrt(np.sum((u_pred - u_ref)**2 + (v_pred - v_ref)**2))
    den = np.sqrt(np.sum(u_ref**2 + v_ref**2))
    return float(num / (den + 1e-12))

def l2_relative_p(p_pred, p_ref):
    num = np.linalg.norm(p_pred - p_ref)
    den = np.linalg.norm(p_ref)
    return float(num / (den + 1e-12))

# -------------------------------
# Função de Avaliação / Métricas
# -------------------------------
def evaluate_metrics(
    pred_df: pd.DataFrame,
    model_metrics: str | None = None,
    out_metrics: str | Path | None = None,
    out_predictions: str | Path | None = None,
) -> dict:
    """
    Calcula métricas de acurácia utilizando um dataframe
    que já contém as predições do modelo.
    """

    required_cols = {
        "Ux", "Uy", "p",
        "Ux_f", "Uy_f", "p_f",
        "dUx", "dUy", "dp",
        "dUx_pred", "dUy_pred", "dp_pred",
        "Ux_corr", "Uy_corr", "p_corr",
    }

    missing = required_cols - set(pred_df.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes: {sorted(missing)}"
        )

    # Campos coarse
    ux_c = pred_df["Ux"].to_numpy(np.float64)
    uy_c = pred_df["Uy"].to_numpy(np.float64)
    p_c = pred_df["p"].to_numpy(np.float64)

    # Campos fine
    ux_f = pred_df["Ux_f"].to_numpy(np.float64)
    uy_f = pred_df["Uy_f"].to_numpy(np.float64)
    p_f = pred_df["p_f"].to_numpy(np.float64)

    # Campos corrigidos
    ux_corr = pred_df["Ux_corr"].to_numpy(np.float64)
    uy_corr = pred_df["Uy_corr"].to_numpy(np.float64)
    p_corr = pred_df["p_corr"].to_numpy(np.float64)

    # Correções verdadeiras
    dux_true = pred_df["dUx"].to_numpy(np.float64)
    duy_true = pred_df["dUy"].to_numpy(np.float64)
    dp_true = pred_df["dp"].to_numpy(np.float64)

    # Correções previstas
    dux_pred = pred_df["dUx_pred"].to_numpy(np.float64)
    duy_pred = pred_df["dUy_pred"].to_numpy(np.float64)
    dp_pred = pred_df["dp_pred"].to_numpy(np.float64)

    # MAE
    mae_coarse = mae_vec(ux_c, uy_c, ux_f, uy_f)
    mae_corr = mae_vec(ux_corr, uy_corr, ux_f, uy_f)

    mae_p_coarse = mae_vec_pressure(p_c, p_f)
    mae_p_corr = mae_vec_pressure(p_corr, p_f)

    mae_deltas = mae_vec(
        dux_pred, duy_pred, dux_true, duy_true
    )
    mae_deltas_p = mae_vec_pressure(dp_pred, dp_true)

    # RMSE
    rmse_uv_coarse = rmse_uv(ux_c, uy_c, ux_f, uy_f)
    rmse_uv_corr = rmse_uv(ux_corr, uy_corr, ux_f, uy_f)

    rmse_p_coarse = rmse_p(p_c, p_f)
    rmse_p_corr = rmse_p(p_corr, p_f)

    # RER
    rer_uv = 1.0 - mae_corr / (mae_coarse + 1e-12)
    rer_p = 1.0 - mae_p_corr / (mae_p_coarse + 1e-12)

    # L2 relativo
    l2_coarse_uv = l2_relative_uv(
        ux_c, uy_c, ux_f, uy_f
    )
    l2_corr_uv = l2_relative_uv(
        ux_corr, uy_corr, ux_f, uy_f
    )

    l2_coarse_p = l2_relative_p(p_c, p_f)
    l2_corr_p = l2_relative_p(p_corr, p_f)

    metrics = {
        "mae_uv_coarse_to_fine": mae_coarse,
        "mae_uv_corrected_to_fine": mae_corr,
        "mae_uv_deltas_pred_vs_true": mae_deltas,
        "mae_p_pressure_coarse_to_fine": mae_p_coarse,
        "mae_p_corrected_to_fine": mae_p_corr,
        "mae_p_deltas_pred_vs_true": mae_deltas_p,

        "rmse_uv_coarse_to_fine": rmse_uv_coarse,
        "rmse_uv_corrected_to_fine": rmse_uv_corr,
        "rmse_p_coarse_to_fine": rmse_p_coarse,
        "rmse_p_corrected_to_fine": rmse_p_corr,

        "l2_rel_uv_coarse": l2_coarse_uv,
        "l2_rel_uv_corrected": l2_corr_uv,
        "l2_rel_p_coarse": l2_coarse_p,
        "l2_rel_p_corrected": l2_corr_p,

        "RER (u,v)": float(rer_uv),
        "RER (p)": float(rer_p),

        "melhora_MAE_pct": float(rer_uv * 100.0),
        "melhora_MAE_pct_pressure": float(rer_p * 100.0),

        "N": int(len(pred_df)),
    }

    label = model_metrics or "MODELO"

    print(f"\n===== MÉTRICAS DE ACURÁCIA ({label}) =====")
    print(f"MAE coarse → fine:     {mae_coarse:.6e}")
    print(f"MAE corrigido → fine:  {mae_corr:.6e}")
    print(f"RMSE coarse → fine:    {rmse_uv_coarse:.6e}")
    print(f"RMSE corrigido → fine: {rmse_uv_corr:.6e}")
    print(f"Melhoria MAE:          {rer_uv * 100:.2f}%")

    if out_metrics is not None:
        out_metrics = Path(out_metrics)
        out_metrics.parent.mkdir(parents=True, exist_ok=True)

        with out_metrics.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if out_predictions is not None:
        out_predictions = Path(out_predictions)
        out_predictions.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(out_predictions, index=False)

    return metrics

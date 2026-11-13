import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from inference import predict_fields
from metrics_physics import evaluate_physics_metrics
from plots import (
    plot_divergence_compare,
    plot_divergence_error,
    plot_divergence_mlp_pinn_compare,
    plot_divergence_error_mlp_pinn,
)
from run_metrics import build_model

def run_physics_metrics_pipeline(
        cfg,
        model_path,
        dataset_full_path,
        xscaler_path,
        yscaler_path,
        metrics_path,
        predictions_path,
        batch_size=4096,
        plots_dir=None,
        comparison_predictions_path=None,
    ):

    # Configuração inicial
    exp_name = cfg['experiment']['name']
    model_type = cfg["model"]["type"]
    feat_cols = cfg["features"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, model_path).to(device)
    xscaler = joblib.load(xscaler_path)
    yscaler = joblib.load(yscaler_path)

    df_full = pd.read_parquet(dataset_full_path)

    required_columns = {
        *feat_cols,
        "x",
        "y",
        "Ux",
        "Uy",
        "p",
        "dUx",
        "dUy",
        "dp",
    }

    missing = required_columns - set(df_full.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset completo: {sorted(missing)}"
        )
    
    # Predição dos campos físicos e divergência
    pred_full = predict_fields(
        model=model,
        df=df_full,
        feat_cols=feat_cols,
        xscaler=xscaler,
        yscaler=yscaler,
        batch_size=batch_size,
    )

    physics_metrics, physics_df = evaluate_physics_metrics(
        pred_df=pred_full,
        n_neighbors=12,
        model_name=exp_name,
    )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            physics_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    physics_df.to_parquet(
        predictions_path,
        index=False,
    )

    generated_plots = {}

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ========================================================
        # Gráficos individuais do modelo atual
        # ========================================================

        plot_specs = {
            "divergence_absolute": {
                "path": plots_dir / f"{model_type}_{exp_name}_divergence_absolute.png",
                "absolute": True,
            },
            "divergence_signed": {
                "path": plots_dir / f"{model_type}_{exp_name}_divergence_signed.png",
                "absolute": False,
            },
        }

        for plot_name, spec in plot_specs.items():
            plot_divergence_compare(
                df_plot=physics_df,
                field_coarse="div_u",
                field_fine="div_u_f",
                field_corrected="div_corrected",
                model_name= model_type,
                absolute=spec["absolute"],
                save_path=spec["path"],
            )

            generated_plots[plot_name] = str(
                spec["path"]
            )

        error_path = (
            plots_dir
            / f"{exp_name}_divergence_error_vs_fine.png"
        )

        plot_divergence_error(
            df_plot=physics_df,
            corrected_field="div_corrected",
            fine_field="div_u_f",
            model_name=model_type,
            save_path=error_path,
        )

        generated_plots[
            "divergence_error_vs_fine"
        ] = str(error_path)

        # ========================================================
        # Comparativo MLP × PINN
        # ========================================================

        if comparison_predictions_path is not None:
            comparison_predictions_path = Path(
                comparison_predictions_path
            )

            if not comparison_predictions_path.exists():
                raise FileNotFoundError(
                    "Arquivo físico do modelo de comparação "
                    f"não encontrado: {comparison_predictions_path}"
                )

            comparison_df = pd.read_parquet(
                comparison_predictions_path
            )

            if model_type == "mlp":
                df_mlp = physics_df
                df_pinn = comparison_df

            elif model_type == "pinn":
                df_mlp = comparison_df
                df_pinn = physics_df

            else:
                raise ValueError(
                    "O comparativo físico suporta apenas "
                    "model_type='mlp' ou model_type='pinn'."
                )

            comparison_absolute_path = (
                plots_dir
                / "divergence_mlp_vs_pinn_absolute.png"
            )

            comparison_signed_path = (
                plots_dir
                / "divergence_mlp_vs_pinn_signed.png"
            )

            comparison_error_path = (
                plots_dir
                / "divergence_error_mlp_vs_pinn.png"
            )

            plot_divergence_mlp_pinn_compare(
                df_mlp=df_mlp,
                df_pinn=df_pinn,
                absolute=True,
                percentile=99.0,
                save_path=comparison_absolute_path,
            )

            plot_divergence_mlp_pinn_compare(
                df_mlp=df_mlp,
                df_pinn=df_pinn,
                absolute=False,
                percentile=99.0,
                save_path=comparison_signed_path,
            )

            plot_divergence_error_mlp_pinn(
                df_mlp=df_mlp,
                df_pinn=df_pinn,
                percentile=99.0,
                save_path=comparison_error_path,
            )

            generated_plots.update({
                "divergence_mlp_vs_pinn_absolute": str(
                    comparison_absolute_path
                ),
                "divergence_mlp_vs_pinn_signed": str(
                    comparison_signed_path
                ),
                "divergence_error_mlp_vs_pinn": str(
                    comparison_error_path
                ),
            })

    return {
        "metrics": physics_metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "plots": generated_plots,
    }
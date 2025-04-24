import os
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback


ENTITY = "erlandpg"
PROJECT = "VFL-CLIP-Attack"


RUN_PATHS = []


offline_datas = [10, 100, 250, 500, 1000]
rounds_to_train_onlines = [0, 10, 50, 99]
whos_attackings = ["image", "text"]
type_attacks = ["attackgradientonline"]


constructed_name = "Attack-SideData{offline_data}_reduce_pNone_r100_or{rounds_to_train_online}_{type_attack}-{who_attacking}"
RUN_NAMES = []


for offline_data in offline_datas:
    for rounds_to_train_online in [99]:
        for who_attacking in ["image"]:
            for type_attack in type_attacks:
                run_name = constructed_name.format(
                    offline_data=offline_data,
                    rounds_to_train_online=rounds_to_train_online,
                    who_attacking=who_attacking,
                    type_attack=type_attack,
                )

                if type_attack == "attackgradientonline":
                    run_name += "FR"
                RUN_NAMES.append(run_name)


CUSTOM_RUN_NAMES = [f"Side Data: {offline_data}" for offline_data in offline_datas]


METRICS_TO_PLOT = [
    "vfl/loss",
    "server_ground_truth/avg_image_embedding_mean",
    "server_ground_truth/avg_image_embedding_std",
    "server_ground_truth/avg_text_embedding_mean",
    "server_ground_truth/avg_text_embedding_std",
    "agg_client_eval/client_prediction/text_emb_mean",
    "agg_client_eval/client_prediction/text_emb_std",
    "agg_client_eval/client_prediction/image_emb_mean",
    "agg_client_eval/client_prediction/image_emb_std",
]


DIFFERENCE_METRICS_CONFIG = [
    {
        "name": "diff/image_mean_embedding_abs",
        "metric_1_key": "server_ground_truth/avg_image_embedding_mean",
        "metric_2_keys": [
            "agg_client_eval/client_prediction/image_emb_mean",
            "client_prediction/image_emb_mean",
        ],
        "plot_type": "absolute",
    },
    {
        "name": "diff/image_std_embedding_abs",
        "metric_1_key": "server_ground_truth/avg_image_embedding_std",
        "metric_2_keys": [
            "agg_client_eval/client_prediction/image_emb_std",
            "client_prediction/image_emb_std",
        ],
        "plot_type": "absolute",
    },
    {
        "name": "diff/text_mean_embedding_abs",
        "metric_1_key": "server_ground_truth/avg_text_embedding_mean",
        "metric_2_keys": [
            "agg_client_eval/client_prediction/text_emb_mean",
            "client_prediction/text_emb_mean",
        ],
        "plot_type": "absolute",
    },
    {
        "name": "diff/text_std_embedding_abs",
        "metric_1_key": "server_ground_truth/avg_text_embedding_std",
        "metric_2_keys": [
            "agg_client_eval/client_prediction/text_emb_std",
            "client_prediction/text_emb_std",
        ],
        "plot_type": "absolute",
    },
]


X_AXIS_KEYS_TO_PLOT = [
    "_step",
]


SMOOTHING_FACTOR = 0.9
SMOOTHING_HALFLIFE_SECONDS = 10


USE_TIME_WEIGHTED_EMA = False


PLOT_GRID = True
PLOT_SAVE_DIR = "plots_attack_comparison"


def smooth_ema(series, factor):
    """Applies Standard Exponential Moving Average smoothing based on data point order."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if not (0 < factor < 1):
        return series
    alpha = 1.0 - factor
    print(
        f"      Applying Standard EWM (Order-Based) using alpha={alpha:.3f} (factor={factor:.3f})..."
    )
    return series.ewm(alpha=alpha, adjust=True, ignore_na=True).mean()


def smooth_ema_time_weighted(series, times, halflife_seconds):
    """Applies Time-Weighted Exponential Moving Average smoothing using timestamps."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if not isinstance(times, pd.Series):
        times = pd.Series(times)
    if not halflife_seconds or halflife_seconds <= 0:
        return series
    df = pd.DataFrame({"data": series, "time": times}).dropna()
    if df.empty:
        print(
            "      WARNING: No valid (data, time) pairs for time-weighted EMA after dropna."
        )
        return pd.Series(index=series.index, dtype=float)
    if not pd.api.types.is_numeric_dtype(df["time"]):
        print(
            "      WARNING: Time values are not numeric. Cannot use time-weighted EMA."
        )
        return series
    try:
        df["time_dt"] = pd.to_datetime(df["time"], unit="s", origin="unix")
    except Exception as e:
        print(
            f"      ERROR: Failed to convert time values to datetime64: {e}. Falling back."
        )
        return series
    if not df["time_dt"].is_monotonic_increasing:
        df = df.sort_values(by="time_dt")
    try:
        halflife_str = f"{halflife_seconds}s"
        print(f"      Applying Time-Weighted EWM using halflife='{halflife_str}'...")
        smoothed_data = (
            df["data"]
            .ewm(
                halflife=halflife_str, times=df["time_dt"], adjust=True, ignore_na=True
            )
            .mean()
        )
    except Exception as e:
        print(
            f"      ERROR: Unexpected error during Time-Weighted EWM calculation: {e}"
        )
        traceback.print_exc()
        return series
    return smoothed_data.reindex(series.index)


def _fetch_metric_history(run, metric_key, x_axis_key):
    """Fetches history for a single metric and the x-axis key."""
    print(f"      Fetching individual history for: ['{metric_key}', '{x_axis_key}']")
    try:
        history = run.history(keys=[metric_key, x_axis_key], pandas=True)
        print(
            f"      Fetched '{metric_key}'. Shape: {history.shape}. Columns: {history.columns.tolist()}"
        )
        if history.empty:
            print(f"      WARNING: History is empty for '{metric_key}'.")
            return None

        history[metric_key] = pd.to_numeric(history[metric_key], errors="coerce")
        history[x_axis_key] = pd.to_numeric(history[x_axis_key], errors="coerce")
        history = history.dropna(subset=[metric_key, x_axis_key])
        if history.empty:
            print(
                f"      WARNING: History became empty for '{metric_key}' after NaN drop."
            )
            return None

        history = history.sort_values(by=x_axis_key).reset_index(drop=True)
        return history
    except wandb.errors.CommError as e:
        print(
            f"      ERROR: Communication error fetching history for '{metric_key}'. Details: {e}"
        )
        return None
    except KeyError:
        print(
            f"      ERROR: Metric key '{metric_key}' not found during individual fetch."
        )
        return None
    except Exception as e:
        print(
            f"      ERROR: Unexpected error fetching history for '{metric_key}'. Details: {e}"
        )
        traceback.print_exc()
        return None


def _plot_single_run_or_diff(
    run,
    metric_key_or_config,
    x_axis_key,
    smoothing_factor,
    time_weighted_smoothing,
    halflife_seconds,
    label_override=None,
):
    """
    Fetches data (handling potential misalignment for difference plots),
    calculates metric/difference, applies smoothing, and plots for one run.
    """
    run_name_for_plot = label_override if label_override is not None else run.name
    print(
        f"    Processing run: '{run.name}' (ID: {run.id}) -> Plot Label: '{run_name_for_plot}'"
    )

    is_difference_plot = isinstance(metric_key_or_config, dict)
    metric_name_for_log = (
        metric_key_or_config["name"] if is_difference_plot else metric_key_or_config
    )
    print(f"      Metric/Task: '{metric_name_for_log}'")

    x_values = None
    y_values_raw = None
    runtime_values = None

    try:
        if is_difference_plot:
            config = metric_key_or_config
            m1_key = config["metric_1_key"]
            m2_keys = config["metric_2_keys"]
            plot_type = config["plot_type"]

            hist1_df = _fetch_metric_history(run, m1_key, x_axis_key)
            if hist1_df is None:
                print(
                    f"      Skipping difference plot due to missing history for '{m1_key}'."
                )
                return False

            hist2_df = None
            m2_key_found = None
            for key in m2_keys:
                hist2_df = _fetch_metric_history(run, key, x_axis_key)
                if hist2_df is not None:
                    m2_key_found = key
                    break

            if hist2_df is None:
                print(
                    f"      Skipping difference plot due to missing history for all potential metric 2 keys: {m2_keys}."
                )
                return False

            print(
                f"      Using '{m1_key}' and '{m2_key_found}' for difference calculation."
            )

            hist1_df = hist1_df.rename(columns={m1_key: "metric1"})
            hist2_df = hist2_df.rename(columns={m2_key_found: "metric2"})

            print(f"      Aligning metrics using '{x_axis_key}' with merge_asof...")

            merged_df = pd.merge_asof(
                hist1_df[[x_axis_key, "metric1"]],
                hist2_df[[x_axis_key, "metric2"]],
                on=x_axis_key,
                direction="nearest",
            )

            merged_df = merged_df.dropna(subset=["metric1", "metric2"])

            if merged_df.empty:
                print(
                    f"      WARNING: No data points remained after aligning '{m1_key}' and '{m2_key_found}' using merge_asof. Steps might be too far apart."
                )
                return False

            print(f"      Alignment complete. Shape after merge: {merged_df.shape}")

            print(f"      Calculating '{plot_type}' difference...")
            if plot_type == "absolute":
                y_values_raw = (merged_df["metric1"] - merged_df["metric2"]).abs()

            else:
                print(f"      ERROR: Unknown plot_type '{plot_type}'. Skipping.")
                return False

            x_values = merged_df[x_axis_key]

            y_values_raw.index = x_values.index

        else:
            metric_key = metric_key_or_config
            history_df = _fetch_metric_history(run, metric_key, x_axis_key)
            if history_df is None:
                print(
                    f"      Skipping standard plot due to missing history for '{metric_key}'."
                )
                return False

            y_values_raw = history_df[metric_key]
            x_values = history_df[x_axis_key]

        runtime_available_and_valid = False
        if time_weighted_smoothing and halflife_seconds and halflife_seconds > 0:
            runtime_df = _fetch_metric_history(run, "_runtime", x_axis_key)
            if runtime_df is not None:
                current_data_df = pd.DataFrame(
                    {"x_orig": x_values, "y_raw": y_values_raw}
                )
                runtime_df = runtime_df.rename(columns={"_runtime": "runtime"})

                aligned_with_runtime_df = pd.merge_asof(
                    current_data_df.sort_values(by="x_orig"),
                    runtime_df[[x_axis_key, "runtime"]].sort_values(by=x_axis_key),
                    left_on="x_orig",
                    right_on=x_axis_key,
                    direction="nearest",
                )
                aligned_with_runtime_df = aligned_with_runtime_df.dropna(
                    subset=["runtime", "y_raw"]
                )

                if not aligned_with_runtime_df.empty:
                    print(
                        f"      Successfully aligned runtime data. Shape: {aligned_with_runtime_df.shape}"
                    )
                    runtime_available_and_valid = True

                    x_values = aligned_with_runtime_df["x_orig"]
                    y_values_raw = aligned_with_runtime_df["y_raw"]
                    runtime_values = aligned_with_runtime_df["runtime"]
                else:
                    print(
                        "      WARNING: Could not align runtime data with metric data using merge_asof. Disabling TWE."
                    )
            else:
                print("      INFO: Could not fetch runtime data. Disabling TWE.")

        if y_values_raw is None or x_values is None:
            print(
                "     ERROR: y_values_raw or x_values is None before smoothing. This shouldn't happen."
            )
            return False

        y_values_smoothed = y_values_raw
        smoothing_applied_type = "None"

        if time_weighted_smoothing:
            if runtime_available_and_valid and runtime_values is not None:
                y_values_smoothed = smooth_ema_time_weighted(
                    y_values_raw, runtime_values, halflife_seconds
                )
                smoothing_applied_type = (
                    f"Time-Weighted (Halflife: {halflife_seconds}s)"
                )
            else:
                pass
        elif smoothing_factor > 0:
            temp_smooth_df = pd.DataFrame(
                {"x": x_values, "y": y_values_raw}
            ).sort_values("x")
            y_values_smoothed = smooth_ema(temp_smooth_df["y"], smoothing_factor)
            y_values_smoothed.index = temp_smooth_df.index
            smoothing_applied_type = f"Standard (Factor: {smoothing_factor:.2f})"

        plot_df = pd.DataFrame({"x": x_values, "y_smooth": y_values_smoothed}).dropna(
            subset=["y_smooth"]
        )

        if plot_df.empty:
            print(
                f"      WARNING: No valid points remain after smoothing (Type: {smoothing_applied_type}). Skipping plot line."
            )
            return False

        plot_df = plot_df.sort_values(by="x")
        plt.plot(
            plot_df["x"], plot_df["y_smooth"], label=run_name_for_plot, linewidth=1.5
        )
        print(
            f"      Plotted data points: {len(plot_df)} (Smoothing: {smoothing_applied_type})"
        )
        return True

    except Exception as e:
        print(
            f"      ERROR: Unexpected error in _plot_single_run_or_diff for run '{run.id}'. Details: {e}"
        )
        traceback.print_exc()
        return False


def _plot_runs_by_path(
    api,
    run_paths,
    custom_run_names,
    metric_key_or_config,
    x_axis_key,
    smoothing_factor,
    time_weighted_smoothing,
    halflife_seconds,
):
    """Plots runs specified by paths, using custom names if provided."""
    print(f"  Plotting {len(run_paths)} runs identified by path...")
    plot_count = 0
    for i, run_path in enumerate(run_paths):
        print(f"\n  Fetching run path: {run_path}")
        try:
            run = api.run(run_path)
            label = (
                custom_run_names[i]
                if custom_run_names and i < len(custom_run_names)
                else None
            )
            if _plot_single_run_or_diff(
                run,
                metric_key_or_config,
                x_axis_key,
                smoothing_factor,
                time_weighted_smoothing,
                halflife_seconds,
                label_override=label,
            ):
                plot_count += 1
        except wandb.errors.CommError as e:
            print(
                f"    ERROR: Could not fetch run '{run_path}'. Check path/permissions. Details: {e}"
            )
        except Exception as e:
            print(
                f"    ERROR: Unexpected error for run path '{run_path}'. Details: {e}"
            )
            traceback.print_exc()
    return plot_count > 0


def _plot_runs_by_name(
    api,
    entity,
    project,
    run_names,
    custom_run_names,
    metric_key_or_config,
    x_axis_key,
    smoothing_factor,
    time_weighted_smoothing,
    halflife_seconds,
):
    """Plots runs specified by names, handling duplicates and custom names."""
    full_project_path = f"{entity}/{project}"
    print(f"  Searching for runs by name in project: {full_project_path}")

    plot_count = 0

    for i, run_name_to_find in enumerate(run_names):
        print(f"\n  Searching for run name: '{run_name_to_find}'...")
        try:
            runs_found = api.runs(
                full_project_path, filters={"display_name": run_name_to_find}
            )

            if not runs_found:
                print(
                    f"    WARNING: No run found with name '{run_name_to_find}'. Skipping."
                )
                continue

            print(f"    Found {len(runs_found)} run(s) with name '{run_name_to_find}'.")
            label_base = (
                custom_run_names[i]
                if custom_run_names and i < len(custom_run_names)
                else None
            )

            for j, run in enumerate(runs_found):
                if len(runs_found) == 1:
                    label = label_base if label_base is not None else run.name
                else:
                    base = label_base or run.name
                    label = f"{base} ({j + 1}/{len(runs_found)}, ID: {run.id})"
                    print(
                        f"      Handling duplicate name instance {j + 1}/{len(runs_found)} (ID: {run.id})"
                    )

                if _plot_single_run_or_diff(
                    run,
                    metric_key_or_config,
                    x_axis_key,
                    smoothing_factor,
                    time_weighted_smoothing,
                    halflife_seconds,
                    label_override=label,
                ):
                    plot_count += 1

        except wandb.errors.CommError as e:
            print(
                f"    ERROR: Could not query runs for project '{full_project_path}'. Details: {e}"
            )
        except Exception as e:
            print(
                f"    ERROR: Unexpected error searching for run name '{run_name_to_find}'. Details: {e}"
            )
            traceback.print_exc()
    return plot_count > 0


def plot_wandb_runs(
    metric_key_or_config,
    x_axis_key="_step",
    run_paths=None,
    entity=None,
    project=None,
    run_names=None,
    custom_run_names=None,
    smoothing_factor=0.0,
    time_weighted_smoothing=False,
    halflife_seconds=None,
    title="W&B Run Comparison",
    xlabel=None,
    ylabel=None,
    grid=True,
    save_dir="plots",
):
    """
    Plots metrics or metric differences from W&B runs, identified by path or name.
    """
    is_difference_plot = isinstance(metric_key_or_config, dict)
    metric_name_for_log = (
        metric_key_or_config["name"] if is_difference_plot else metric_key_or_config
    )

    print(
        f"\n--- Generating Plot: '{title}' (Task: {metric_name_for_log}, X-axis: {x_axis_key}) ---"
    )

    use_path_method = bool(run_paths)
    use_name_method = bool(entity and project and run_names)

    if not use_path_method and not use_name_method:
        print("ERROR: No runs specified.")
        return

    if ylabel is None:
        smooth_desc = ""

        if time_weighted_smoothing and halflife_seconds and halflife_seconds > 0:
            smooth_desc = f" (Smoothed, {halflife_seconds}s Halflife)"

        elif (
            not time_weighted_smoothing
            and smoothing_factor > 0
            and smoothing_factor < 1
        ):
            smooth_desc = f" (Smoothed, Factor {smoothing_factor:.2f})"

        if is_difference_plot:
            config = metric_key_or_config

            try:
                m1_simple = (
                    config["metric_1_key"]
                    .split("/")[-1]
                    .replace("avg_", "")
                    .replace("embedding_", "")
                    .replace("_", " ")
                    .strip()
                )

                m2_simple = (
                    config["metric_2_keys"][0]
                    .split("/")[-1]
                    .replace("client_prediction/", "")
                    .replace("_emb", "")
                    .replace("_", " ")
                    .strip()
                )
                diff_type_desc = config.get("plot_type", "Difference").title()
                ylabel = f"{diff_type_desc} ({m1_simple.title()} vs {m2_simple.title()}){smooth_desc}"
            except Exception:
                ylabel = f"{metric_name_for_log.replace('_', ' ').title()}{smooth_desc}"
        else:
            ylabel = f"{metric_key_or_config.replace('_', ' ').title()}{smooth_desc}"

    if xlabel is None:
        if x_axis_key == "_step":
            xlabel = "Steps"
        elif x_axis_key == "_runtime":
            xlabel = "Runtime (seconds)"
        elif x_axis_key == "_timestamp":
            xlabel = "Timestamp"
        else:
            xlabel = x_axis_key.replace("_", " ").title()

    print("  Initializing W&B API...")
    try:
        api = wandb.Api(timeout=60)
        print("  API initialized.")
    except Exception as e:
        print(f"FATAL: Failed to initialize W&B API: {e}")
        traceback.print_exc()
        return

    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plot_successful = False
    if use_path_method:
        plot_successful = _plot_runs_by_path(
            api,
            run_paths,
            custom_run_names,
            metric_key_or_config,
            x_axis_key,
            smoothing_factor,
            time_weighted_smoothing,
            halflife_seconds,
        )
    elif use_name_method:
        plot_successful = _plot_runs_by_name(
            api,
            entity,
            project,
            run_names,
            custom_run_names,
            metric_key_or_config,
            x_axis_key,
            smoothing_factor,
            time_weighted_smoothing,
            halflife_seconds,
        )

    print("\n  Finalizing plot...")
    if plot_successful and len(plt.gca().lines) > 0:
        num_lines = len(plt.gca().lines)
        legend_fontsize = 10 if num_lines <= 10 else 8 if num_lines <= 20 else 6
        plt.legend(fontsize=legend_fontsize, loc="best")
        if grid:
            plt.grid(True, linestyle="--", alpha=0.6)

        os.makedirs(save_dir, exist_ok=True)
        filename_metric = metric_name_for_log.replace("/", "_").replace(" ", "_")
        filename_xaxis = x_axis_key.lstrip("_")
        save_path = os.path.join(
            save_dir, f"wandb_compare_{filename_metric}_vs_{filename_xaxis}.png"
        )

        try:
            plt.tight_layout()
            print(f"  Saving plot to '{save_path}'.")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print("--- Plot Generation Complete ---")
        except Exception as e:
            print(f"  ERROR: Failed to save plot to '{save_path}'. Details: {e}")
            traceback.print_exc()
            plt.close()
    else:
        print("  WARNING: No data was successfully plotted for this configuration.")
        plt.close()
        print("--- Plot Generation Skipped ---")


if __name__ == "__main__":
    print("Validating configuration...")
    method1_configured = bool(RUN_PATHS)
    method2_configured = (
        bool(ENTITY and PROJECT and RUN_NAMES)
        and ENTITY != "your_entity"
        and PROJECT != "your_project"
        and len(RUN_NAMES) > 0
    )
    valid_config = True
    if not method1_configured and not method2_configured:
        print("\n" + "=" * 30 + " CONFIGURATION ERROR " + "=" * 30)
        print(
            "Please configure EITHER non-empty RUN_PATHS OR (ENTITY, PROJECT, and non-empty RUN_NAMES)."
        )
        print("=" * 80 + "\n")
        valid_config = False
    elif method1_configured and method2_configured:
        print("\n" + "=" * 30 + " CONFIGURATION WARNING " + "=" * 30)
        print("Both RUN_PATHS and RUN_NAMES are configured. Using RUN_PATHS.")
        print("=" * 80 + "\n")
        method2_configured = False

    if valid_config and CUSTOM_RUN_NAMES:
        expected_len = len(RUN_PATHS) if method1_configured else len(RUN_NAMES)
        if len(CUSTOM_RUN_NAMES) != expected_len:
            print("\n" + "=" * 30 + " CONFIGURATION ERROR " + "=" * 30)
            print(
                f"Length mismatch: CUSTOM_RUN_NAMES ({len(CUSTOM_RUN_NAMES)}) vs specified runs ({expected_len})."
            )
            print("=" * 80 + "\n")
            valid_config = False

    if (
        valid_config
        and USE_TIME_WEIGHTED_EMA
        and (not SMOOTHING_HALFLIFE_SECONDS or SMOOTHING_HALFLIFE_SECONDS <= 0)
    ):
        print("\n" + "=" * 30 + " CONFIGURATION WARNING " + "=" * 30)
        print(
            f"USE_TIME_WEIGHTED_EMA is True, but SMOOTHING_HALFLIFE_SECONDS ({SMOOTHING_HALFLIFE_SECONDS}) is invalid. TWE will be disabled."
        )
        print("=" * 80 + "\n")
    if valid_config and not USE_TIME_WEIGHTED_EMA and not (0 <= SMOOTHING_FACTOR < 1):
        print("\n" + "=" * 30 + " CONFIGURATION WARNING " + "=" * 30)
        print(
            f"USE_TIME_WEIGHTED_EMA is False, but SMOOTHING_FACTOR ({SMOOTHING_FACTOR}) is invalid. Standard smoothing will be disabled."
        )
        print("=" * 80 + "\n")

    if valid_config:
        print("\nStarting plot generation process...")
        custom_names_to_pass = CUSTOM_RUN_NAMES
        plot_tasks = []
        for metric in METRICS_TO_PLOT:
            plot_tasks.append({"type": "standard", "metric_key": metric})
        for diff_config in DIFFERENCE_METRICS_CONFIG:
            plot_tasks.append({"type": "difference", "config": diff_config})

        for task in plot_tasks:
            for x_key in X_AXIS_KEYS_TO_PLOT:
                if task["type"] == "standard":
                    metric_key = task["metric_key"]
                    plot_title_base = metric_key.replace("_", " ").title()
                    metric_key_or_config = metric_key
                else:
                    config = task["config"]

                    plot_title_base = (
                        config["name"]
                        .replace("diff/", "Difference: ")
                        .replace("_", " ")
                        .title()
                    )
                    metric_key_or_config = config

                x_axis_title = (
                    "Runtime"
                    if x_key == "_runtime"
                    else x_key.replace("_", " ").title()
                )
                plot_title = f"{plot_title_base} vs {x_axis_title}"

                plot_wandb_runs(
                    metric_key_or_config=metric_key_or_config,
                    x_axis_key=x_key,
                    run_paths=RUN_PATHS if method1_configured else None,
                    entity=ENTITY if method2_configured else None,
                    project=PROJECT if method2_configured else None,
                    run_names=RUN_NAMES if method2_configured else None,
                    custom_run_names=custom_names_to_pass,
                    smoothing_factor=SMOOTHING_FACTOR,
                    time_weighted_smoothing=USE_TIME_WEIGHTED_EMA,
                    halflife_seconds=SMOOTHING_HALFLIFE_SECONDS,
                    title=plot_title,
                    grid=PLOT_GRID,
                    save_dir=PLOT_SAVE_DIR,
                )
        print("\nAll requested plot generations attempted.")

    runtime_calc_possible = (
        valid_config
        and method2_configured
        and len(RUN_NAMES) == 2
        and "_runtime" in X_AXIS_KEYS_TO_PLOT
    )
    if runtime_calc_possible:
        print("\n" + "=" * 30 + " RUNTIME MULTIPLICATION CALCULATION " + "=" * 30)

        print("=" * 80 + "\n")
    elif valid_config:
        if not (method2_configured and len(RUN_NAMES) == 2):
            print(
                "\n"
                + "=" * 30
                + " RUNTIME MULTIPLICATION SKIPPED (Requires exactly 2 RUN_NAMES) "
                + "=" * 30
                + "\n"
            )
        elif "_runtime" not in X_AXIS_KEYS_TO_PLOT:
            print(
                "\n"
                + "=" * 30
                + " RUNTIME MULTIPLICATION SKIPPED ('_runtime' not in X_AXIS_KEYS_TO_PLOT) "
                + "=" * 30
                + "\n"
            )

    else:
        print(
            "\nPlot generation and runtime calculation skipped due to configuration errors."
        )

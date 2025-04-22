import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

# --- Configuration ---
WANDB_ENTITY = "erlandpg" # Default Entity
WANDB_PROJECT = "VFL-CLIP-Attack"   # Default Project

# --- Helper Functions ---

def fetch_run_data(api, entity, project, run_id):
    """Fetches historical metrics for a specific run ID as a Pandas DataFrame."""
    try:
        print(f"Fetching run: {entity}/{project}/{run_id}")
        run = api.run(f"{entity}/{project}/{run_id}")
        # Fetch all keys initially, then filter. Or specify keys if you know them all.
        # Increase timeout for potentially large histories
        history = run.history(pandas=True, stream="default", samples=run.summary.get("_step", 5000) * 2) # Fetch more samples

        # Ensure '_step' is the index for easy plotting against rounds/steps
        if '_step' in history.columns:
            history = history.set_index('_step').sort_index() # Sort by step
             # Drop duplicate steps if any, keeping the last entry
            history = history[~history.index.duplicated(keep='last')]
        elif history.index.name == '_step':
             history = history.sort_index() # Already indexed by step
             history = history[~history.index.duplicated(keep='last')]
        else:
             print(f"Warning: '_step' column not found or not index for run {run_id}. Using default index.")

        print(f"Successfully fetched data for run: {run.name} ({run_id}) with {len(history)} steps.")
        # print("Available columns:", history.columns.tolist()) # Debug: List columns
        return run, history
    except Exception as e:
        print(f"Error fetching data for run {run_id} (Entity: {entity}, Project: {project}): {e}")
        return None, None

def get_comparison_keys(run_config, history_df):
    """Determines the metric keys to compare based on who was attacking."""
    # --- Determine Attacker Type ---
    # whos_attacking = run_config.get("attack_whos_attacking", run_config.get("whos_attacking", "text"))
    whos_attacking = "text"
    print(f"Determined attacker type: {whos_attacking}")

    # --- Determine Client Namespace ---
    # Find columns matching the client prediction pattern
    client_pred_cols = [col for col in history_df.columns if '/prediction/' in col]
    attacker_cid = None
    pred_namespace = None

    if whos_attacking == "image":
        attack_target = "Text Embeddings"
        # Look for namespaces like client_0, client_2, ...
        possible_cids = range(0, 20, 2) # Check even IDs
    elif whos_attacking == "text":
        attack_target = "Image Embeddings"
        # Look for namespaces like client_1, client_3, ...
        possible_cids = range(1, 20, 2) # Check odd IDs
    else:
         raise ValueError(f"Unknown attacker type in run config: {whos_attacking}")

    # Find the first matching namespace/CID from the actual columns
    for cid in possible_cids:
        prefix = f"client_prediction"
        if any(col.startswith(prefix) for col in client_pred_cols):
             pred_namespace = f"client_prediction"
             attacker_cid = cid
             print(f"Found prediction namespace: {pred_namespace}, Attacker CID: {attacker_cid}")
             break

    if attacker_cid is None:
         print(f"Warning: Could not automatically determine prediction namespace/CID for attacker '{whos_attacking}'. Falling back to default assumptions (CID 0 for image, 1 for text).")
         attacker_cid = 0 if whos_attacking == 'image' else 1
         pred_namespace = f"client_prediction"


    # --- Define Keys Based on Attacker ---
    if whos_attacking == "image":
        pred_mean_key = f"{pred_namespace}/text_emb_mean"
        pred_std_key = f"{pred_namespace}/text_emb_std"
        gt_mean_key = "server_ground_truth/avg_text_embedding_mean"
        gt_std_key = "server_ground_truth/avg_text_embedding_std"
    else: # whos_attacking == "text"
        pred_mean_key = f"{pred_namespace}/image_emb_mean"
        pred_std_key = f"{pred_namespace}/image_emb_std"
        gt_mean_key = "server_ground_truth/avg_image_embedding_mean"
        gt_std_key = "server_ground_truth/avg_image_embedding_std"


    print(f"Using Keys: PredMean='{pred_mean_key}', PredStd='{pred_std_key}', GTMean='{gt_mean_key}', GTStd='{gt_std_key}'")

    return {
        "pred_mean": pred_mean_key,
        "pred_std": pred_std_key,
        "gt_mean": gt_mean_key,
        "gt_std": gt_std_key,
        "target": attack_target,
        "attacker_cid": attacker_cid # Return the determined or default CID
    }


def plot_single_comparison(
    plot_df, rounds, key1, key2, key1_ema, key2_ema, # Add EMA data
    label1, label2, ylabel, title, filename, args):
    """Helper function to create and save a single comparison plot with optional EMA."""
    fig, ax = plt.figure(figsize=args.figsize), plt.gca()

    # Plot raw data
    ax.plot(rounds, plot_df[key1], label=label1, marker='.', linestyle='-',
             linewidth=args.linewidth, markersize=args.markersize, alpha=0.6) # Make raw data slightly transparent
    ax.plot(rounds, plot_df[key2], label=label2, marker='x', linestyle='--',
             linewidth=args.linewidth, markersize=args.markersize, alpha=0.6) # Make raw data slightly transparent

    # Plot EMA data if available
    if args.ema_span > 0 and key1_ema is not None and key2_ema is not None:
        ax.plot(rounds, key1_ema, label=f'{label1} (EMA)', linestyle='-',
                 linewidth=args.linewidth + 0.5) # Slightly thicker EMA line
        ax.plot(rounds, key2_ema, label=f'{label2} (EMA)', linestyle='--',
                 linewidth=args.linewidth + 0.5) # Slightly thicker EMA line


    ax.set_ylabel(ylabel, fontsize=args.label_fontsize)
    ax.set_title(title, fontsize=args.title_fontsize)
    ax.set_xlabel("Server Round (_step)", fontsize=args.label_fontsize)
    ax.legend(fontsize=args.legend_fontsize)
    ax.grid(True)
    plt.xticks(fontsize=args.tick_fontsize)
    plt.yticks(fontsize=args.tick_fontsize)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close(fig) # Close the figure to free memory

def plot_single_difference(
    plot_df, rounds, diff_data, diff_data_ema, # Add EMA data
    label, ylabel, title, filename, args):
    """Helper function to create and save a single difference plot with optional EMA."""
    fig, ax = plt.figure(figsize=args.figsize), plt.gca()

    # Plot raw difference
    ax.plot(rounds, diff_data, label=label, marker='o', linestyle=':',
             linewidth=args.linewidth, markersize=args.markersize, alpha=0.6) # Make raw data slightly transparent

    # Plot EMA difference if available
    if args.ema_span > 0 and diff_data_ema is not None:
         ax.plot(rounds, diff_data_ema, label=f'{label} (EMA)', linestyle='-', # Solid line for EMA diff
                 linewidth=args.linewidth + 0.5) # Slightly thicker EMA line

    ax.set_ylabel(ylabel, fontsize=args.label_fontsize)
    ax.set_title(title, fontsize=args.title_fontsize)
    ax.set_xlabel("Server Round (_step)", fontsize=args.label_fontsize)
    ax.legend(fontsize=args.legend_fontsize)
    ax.grid(True)
    # Optional: Set y-axis limit starting from 0 for difference plots
    ax.set_ylim(bottom=0)
    plt.xticks(fontsize=args.tick_fontsize)
    plt.yticks(fontsize=args.tick_fontsize)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close(fig) # Close the figure to free memory


def plot_attack_comparison(run, history_df, keys, args):
    """Creates and saves separate comparison plots for a single run, with optional EMA."""
    run_name = run.name
    run_id = run.id
    save_dir = args.plot_dir

    pred_mean_key = keys["pred_mean"]
    pred_std_key = keys["pred_std"]
    gt_mean_key = keys["gt_mean"]
    gt_std_key = keys["gt_std"]
    attack_target = keys["target"]
    attacker_cid = keys["attacker_cid"] # Get attacker CID

    # Check if necessary keys exist in the DataFrame
    required_keys = [pred_mean_key, pred_std_key, gt_mean_key, gt_std_key]
    missing_keys = [k for k in required_keys if k not in history_df.columns]
    if missing_keys:
        print(f"Warning: Run '{run_name}' ({run_id}) is missing keys: {missing_keys}. Skipping plots.")
        return

    # Drop rows where any of the key metrics are NaN
    # Important: Do this *before* calculating EMA
    plot_df = history_df[required_keys].dropna()
    if plot_df.empty or len(plot_df) < 2: # Need at least 2 points for EMA/plotting
        print(f"Warning: Run '{run_name}' ({run_id}) has insufficient complete data ({len(plot_df)} points) for plotting after dropping NaNs. Skipping.")
        return

    rounds = plot_df.index # Use the '_step' index
    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists

    # --- Calculate EMA if requested ---
    gt_mean_ema, pred_mean_ema, gt_std_ema, pred_std_ema = None, None, None, None
    if args.ema_span > 0:
        print(f"Calculating EMA with span={args.ema_span}...")
        # Use adjust=False for standard EMA behavior
        gt_mean_ema = plot_df[gt_mean_key].ewm(span=args.ema_span, adjust=False).mean()
        pred_mean_ema = plot_df[pred_mean_key].ewm(span=args.ema_span, adjust=False).mean()
        gt_std_ema = plot_df[gt_std_key].ewm(span=args.ema_span, adjust=False).mean()
        pred_std_ema = plot_df[pred_std_key].ewm(span=args.ema_span, adjust=False).mean()


    # --- Plot 1: Mean Comparison ---
    plot_single_comparison(
        plot_df, rounds,
        gt_mean_key, pred_mean_key,
        gt_mean_ema, pred_mean_ema, # Pass EMA data
        label1=f'Ground Truth Mean ({gt_mean_key.split("/")[-1]})',
        label2=f'Predicted Mean (Client {attacker_cid})',
        ylabel="Embedding Mean",
        title=f"Mean Value Comparison: {attack_target}\nRun: {run_name}",
        filename=os.path.join(save_dir, f"attack_eval_mean_{run_id}.png"),
        args=args
    )

    # --- Plot 2: Standard Deviation Comparison ---
    plot_single_comparison(
        plot_df, rounds,
        gt_std_key, pred_std_key,
        gt_std_ema, pred_std_ema, # Pass EMA data
        label1=f'Ground Truth Std Dev ({gt_std_key.split("/")[-1]})',
        label2=f'Predicted Std Dev (Client {attacker_cid})',
        ylabel="Embedding Std Dev",
        title=f"Std Dev Comparison: {attack_target}\nRun: {run_name}",
        filename=os.path.join(save_dir, f"attack_eval_std_{run_id}.png"),
        args=args
    )

    # --- Calculate Differences ---
    mean_diff = np.abs(plot_df[gt_mean_key] - plot_df[pred_mean_key])
    std_diff = np.abs(plot_df[gt_std_key] - plot_df[pred_std_key])

    # --- Calculate EMA for Differences if requested ---
    mean_diff_ema, std_diff_ema = None, None
    if args.ema_span > 0:
        mean_diff_ema = mean_diff.ewm(span=args.ema_span, adjust=False).mean()
        std_diff_ema = std_diff.ewm(span=args.ema_span, adjust=False).mean()


    # --- Plot 3a: Absolute Difference in Mean ---
    plot_single_difference(
        plot_df, rounds, mean_diff, mean_diff_ema, # Pass EMA data
        label='Abs. Difference in Mean',
        ylabel="Absolute Difference",
        title=f"Absolute Difference in Mean: {attack_target}\nRun: {run_name}",
        filename=os.path.join(save_dir, f"attack_eval_diff_mean_{run_id}.png"),
        args=args
    )
    # --- Plot 3b: Absolute Difference in Std Dev ---
    plot_single_difference(
        plot_df, rounds, std_diff, std_diff_ema, # Pass EMA data
        label='Abs. Difference in Std Dev',
        ylabel="Absolute Difference",
        title=f"Absolute Difference in Std Dev: {attack_target}\nRun: {run_name}",
        filename=os.path.join(save_dir, f"attack_eval_diff_std_{run_id}.png"),
        args=args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VFL Attack using W&B logs and create separate plots with optional EMA smoothing.")
    parser.add_argument("run_ids", nargs='+', help="One or more W&B run IDs to evaluate (e.g., abc123xyz def456uvw)")
    parser.add_argument("--entity", type=str, default=WANDB_ENTITY, help="W&B entity (username or team)")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--plot_dir", type=str, default="attack_plots", help="Directory to save output plots")

    # --- Arguments for customization ---
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6], help="Figure size (width height) in inches") # Adjusted default size
    parser.add_argument("--title_fontsize", type=int, default=14, help="Font size for plot titles")
    parser.add_argument("--label_fontsize", type=int, default=12, help="Font size for axis labels")
    parser.add_argument("--legend_fontsize", type=int, default=10, help="Font size for legends")
    parser.add_argument("--tick_fontsize", type=int, default=10, help="Font size for axis ticks")
    parser.add_argument("--linewidth", type=float, default=1.5, help="Line width for plots")
    parser.add_argument("--markersize", type=float, default=4, help="Marker size for plots")
    # --- EMA Argument ---
    parser.add_argument("--ema_span", type=int, default=10, help="Span for EMA smoothing (e.g., 10). Set to 0 to disable EMA.")


    args = parser.parse_args()

    # Ensure entity is set
    if not args.entity:
        print("Error: W&B entity not set. Use --entity or set WANDB_ENTITY in the script.")
        exit(1)

    # Initialize W&B API
    try:
        api = wandb.Api(timeout=19) # Increased timeout for API calls
    except Exception as e:
        print(f"Error initializing W&B API. Have you logged in (`wandb login`)? Error: {e}")
        exit(1)

    # Process each run ID provided
    for run_id in args.run_ids:
        print(f"\n--- Processing Run ID: {run_id} ---")
        run, history_df = fetch_run_data(api, args.entity, args.project, run_id)

        if run and history_df is not None and not history_df.empty:
            try:
                # Get the comparison keys based on the run's config AND history columns
                comparison_keys = get_comparison_keys(run.config, history_df)
                # Pass all args to the plotting function
                plot_attack_comparison(run, history_df, comparison_keys, args)
            except ValueError as e:
                print(f"Skipping plot for run {run_id}: {e}")
            except KeyError as e:
                 print(f"Skipping plot for run {run_id}: Missing expected data key in history: {e}. Available columns: {list(history_df.columns)}")
            except Exception as e:
                 print(f"An unexpected error occurred while plotting run {run_id}: {e}")
                 # import traceback # Uncomment for detailed debug
                 # traceback.print_exc() # Uncomment for detailed debug
        else:
            print(f"Skipping analysis for run {run_id} due to fetch errors or empty history.")

    print("\nEvaluation complete.")
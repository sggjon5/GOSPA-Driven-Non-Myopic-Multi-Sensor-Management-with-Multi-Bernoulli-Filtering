import numpy as np

npz_path = r"C:\Users\11geo\Desktop\PhD Corrections\IEEE TAES Code and Results End of PhD - Non myopic KLD added\results\GOSPA\Clutter 2\mcts4_errors.npz" # change to your file
   # <- change this if your file is elsewhere
d = np.load(npz_path, allow_pickle=True)

def mc_table_stats(arr_sq: np.ndarray):
    """
    arr_sq: shape (n_runs, n_timesteps), stored as squared (p=2) errors.
    Returns: (mean_across_runs, std_across_runs, per_run_scores)
    where per_run_scores[r] = mean_t( sqrt(arr_sq[r,t]) )
    """
    rmse = np.sqrt(arr_sq.astype(float))          # convert to RMS scale
    per_run_scores = rmse.mean(axis=1)            # one scalar score per run
    mean_score = float(per_run_scores.mean())
    std_score = float(per_run_scores.std(ddof=1)) if per_run_scores.size > 1 else 0.0
    return mean_score, std_score, per_run_scores

# Compute table stats for each metric
metrics = ["gospa", "localisation", "missed", "false"]
results = {}

for m in metrics:
    mean_m, std_m, per_run = mc_table_stats(d[m])
    results[m] = (mean_m, std_m)
    print(f"{m:13s}  mean = {mean_m:.6f}   std = {std_m:.6f}   (runs={per_run.size})")

# If you want: save per-run scores to CSV for inspection
# (creates one CSV per metric)
# out_dir = "/mnt/data"
# for m in metrics:
#     _, _, per_run = mc_table_stats(d[m])
#     # np.savetxt(f"{out_dir}/MCTS1_{m}_per_run_scores.csv",
#     #            np.column_stack([np.arange(per_run.size), per_run]),
#     #            delimiter=",",
#     #            header="mc_run,score_mean_over_time_RMS",
#     #            comments="")
# print("\nSaved per-run score CSVs to:", out_dir)


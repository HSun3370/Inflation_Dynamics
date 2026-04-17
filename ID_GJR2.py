
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from BEGE_GARCH import ID_GARCH

import os
  
parser = argparse.ArgumentParser(description="seed sets")

parser.add_argument("--id", type=int, default=1) 
args = parser.parse_args()  # <<< after all arguments are added
seed = args.id  

 

 
# Load data
sample_data = pd.read_pickle('/project/lhansen/Capital_NN_variant/BEGE_GARCH/Aggregate_CPI_inflation.pkl')

# Base directory to store results
base_dir = "/project/lhansen/Capital_NN_variant/BEGE_GARCH/RandomDraw_ID"
# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

 
 
spec = {
        "mean_type": "ARX(1,1)",
        "Y": sample_data["Inflation"],
        "X": sample_data["Forecasted inflation"],
        "folder_name":"ARX11"
    }


# for spec in model_specs:

mean_type = spec["mean_type"]
foldername= spec["folder_name"]
# Use the mean_type as the subfolder name
out_dir = os.path.join(base_dir, foldername)
os.makedirs(out_dir, exist_ok=True)

 
# ==============================================================
# Robust incremental saving: all results go into ONE pkl file
# ==============================================================

out_file = os.path.join(out_dir, f"draw_{seed}.pkl")

# If the file already exists, resume from where you left off
if os.path.exists(out_file):
    container = pd.read_pickle(out_file)
    start_iter = len(container) + 1
    print(f"Resuming from iteration {start_iter} (already have {len(container)} results)")
else:
    container = []
    start_iter = 1
    print("Starting fresh run")

# Loop through iterations
for i in range(start_iter, 51):  # or 501
    try:
        res = ID_GARCH(
            Y=spec["Y"],
            X=spec["X"],
            mean_type=mean_type,
            n_starts=50,
            maxiter=500,
            tol=1e-8,
            random_state=i + seed * 10000,
        )

        container.append(res)
        pd.to_pickle(container, out_file)  # overwrite with latest list

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Saved iteration {i}/50 to {out_file} (total {len(container)} results)")

    except Exception as e:
        # Print error message but keep looping
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Error at iteration {i}: {e}. Skipping to next iteration.")
        continue

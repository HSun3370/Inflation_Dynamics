
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from BEGE_GARCH import BEGE_Constant_MLE

import os
  
parser = argparse.ArgumentParser(description="seed sets")

parser.add_argument("--id", type=int, default=1) 
args = parser.parse_args()  # <<< after all arguments are added
seed = args.id  

 

 
# Load data
sample_data = pd.read_pickle('/project/lhansen/Capital_NN_variant/BEGE_GARCH/Aggregate_CPI_inflation.pkl')

# Base directory to store results
base_dir = "/project/lhansen/Capital_NN_variant/BEGE_GARCH/RandomDraw_Constant"
# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# Define model specs (use mean_type as folder name)
model_specs = [
    {
        "mean_type": "constant",
        "Y": sample_data["Inflation shock"],
        "X": None, 
        "folder_name":"constant"
    },
    {
        "mean_type": "ARX(1,1)",
        "Y": sample_data["Inflation"],
        "X": sample_data["Forecasted inflation"],
        "folder_name":"ARX11"
    },
    {
        "mean_type": "ARX(2,1)",
        "Y": sample_data["Inflation"],
        "X": sample_data["Forecasted inflation"],
        "folder_name":"ARX21"
    },
    {
        "mean_type": "ARX(2,2)",
        "Y": sample_data["Inflation"],
        "X": sample_data["Forecasted inflation"],
        "folder_name":"ARX22"
    },
]
 
for spec in model_specs:
    mean_type = spec["mean_type"]
    foldername= spec["folder_name"]
    # Use the mean_type as the subfolder name
    out_dir = os.path.join(base_dir, foldername)
    os.makedirs(out_dir, exist_ok=True)

    container = []  # one container per model
    # Loop 500 times and set random_state using i
    for i in range(1, 501):  # 1..500; change to range(500) if you prefer 0..499
        res = BEGE_Constant_MLE(
            Y=spec["Y"],
            X=spec["X"],
            mean_type=mean_type,
            n_starts=20,
            maxiter=1500,
            tol=1e-8,
            random_state=i+seed*10000,   # <- seed per iteration
        )
        # res is expected to be a dictionary; append to the container
        container.append(res)

    # Save this model's container to its own file
    out_file = os.path.join(out_dir, f"draw_{seed}.pkl")
    pd.to_pickle(container, out_file)
    print(f"Saved {mean_type} results ({len(container)} draws) to: {out_file}")
"""
Store functions to generate inputs and outputs
"""
import os
import pandas as pd

def save_travel_path(
    travel_paths: str,
    folder_name: str,
) -> None:
    travel_paths = pd.DataFrame(
        travel_paths,
        columns=["person_idx", "role", "travel_path"]
    )
    fn = os.path.join(folder_name, "trip_paths.csv")
    with open(fn, 'a') as f:
        travel_paths.to_csv(f, index=False, mode='a', header=f.tell() == 0)

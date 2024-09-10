import pickle
import os
from pathlib import Path

# Constants
DATA_ROOT = Path("data/dair_vic_kitti_format")

# File names
PKL_FILES = {
    "1214": {
        "trainval": "dair_coop1214_infos_trainval.pkl",
        "train": "dair_coop1214_infos_train.pkl",
        "val": "dair_coop1214_infos_val.pkl",
    },
    "c": {
        "trainval": "dair_vic_kitti_format_infos_trainval.pkl",
        "train": "dair_vic_kitti_format_infos_train.pkl",
        "val": "dair_vic_kitti_format_infos_val.pkl",
    },
}


def load_pickle(file_path):
    """Load pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    """Save data to pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# Load data
data = {
    "1214": {k: load_pickle(DATA_ROOT / v) for k, v in PKL_FILES["1214"].items()},
    "c": {k: load_pickle(DATA_ROOT / v) for k, v in PKL_FILES["c"].items()},
}


def process_and_save_data(data_1214, data_c, dataset_name):
    """
    Print information about images in data_1214 that are not in data_c,
    delete those entries, and save the updated data_1214.
    """
    data_c_idx_set = {d["image"]["image_idx"] for d in data_c}
    new_data_1214 = []
    deleted_count = 0

    for d in data_1214:
        if d["image"]["image_idx"] not in data_c_idx_set:
            print(f"Deleting - Image idx: {d['image']['image_idx']}")
            print(
                f"Image path: {d['image']['image_path']}, "  # exists: {Path(os.path.join(DATA_ROOT,Path(d['image']['image_path']))).exists()},
            )
            print(
                f"Inf Image: {d['image']['inf_image_path']}, "
            )  # path exists: {Path(os.path.join(DATA_ROOT,Path(d['image']['inf_image_path']))).exists()},
            print()
            deleted_count += 1
        else:
            new_data_1214.append(d)

    print(f"Deleted {deleted_count} entries from {dataset_name} Set")
    print(f"Original size: {len(data_1214)}, New size: {len(new_data_1214)}")

    # Save updated data
    save_path = DATA_ROOT / f"updated_{PKL_FILES['1214'][dataset_name.lower()]}"
    # save_pickle(new_data_1214, save_path)
    print(f"Updated data saved to {save_path}")

    return new_data_1214


# Process and save updated data for each dataset
updated_data = {}
for dataset in ["trainval", "train", "val"]:
    print(f"\n{dataset.capitalize()} Set:")
    print(
        f"Original: {len(data['1214'][dataset])}, Reference: {len(data['c'][dataset])}"
    )
    print()
    updated_data[dataset] = process_and_save_data(
        data["1214"][dataset], data["c"][dataset], dataset.capitalize()
    )

# Print final statistics
print("\nFinal Statistics:")
for dataset in ["trainval", "train", "val"]:
    print(
        f"{dataset.capitalize()} Set: Original: {len(data['1214'][dataset])}, Updated: {len(updated_data[dataset])}, Reference: {len(data['c'][dataset])}"
    )

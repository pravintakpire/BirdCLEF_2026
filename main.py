import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import load_config, ROOT, OUTPUTS_DIR, SUBMISSIONS_DIR
from src.dataset import BirdCLEFDataset
from src.models import BirdCLEFModel
from src.train import run_training
from src.inference import generate_submission
from src.utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "train":
        train_csv = ROOT / config["train_csv"]
        audio_dir = ROOT / config["train_audio_dir"]
        output_dir = ROOT / config["output_dir"]

        df = pd.read_csv(train_csv)
        species_list = sorted(df["primary_label"].unique().tolist())
        print(f"Found {len(species_list)} species")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])
        splits = list(skf.split(df, df["primary_label"]))
        train_idx, val_idx = splits[config["val_fold"]]

        mel_cache_dir = ROOT / config["mel_cache_dir"] if config.get("mel_cache_dir") else None
        train_ds = BirdCLEFDataset(df.iloc[train_idx], audio_dir, species_list, config, train=True, mel_cache_dir=mel_cache_dir)
        val_ds = BirdCLEFDataset(df.iloc[val_idx], audio_dir, species_list, config, train=False, mel_cache_dir=mel_cache_dir)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        model = BirdCLEFModel(num_classes=len(species_list), model_name=config["model_name"], pretrained=config["pretrained"])
        model = model.to(device)

        run_training(model, train_loader, val_loader, config, output_dir, device)

    elif args.mode == "infer":
        train_csv = ROOT / config["train_csv"]
        df = pd.read_csv(train_csv)
        species_list = sorted(df["primary_label"].unique().tolist())

        model = BirdCLEFModel(num_classes=len(species_list), model_name=config["model_name"], pretrained=False)
        checkpoint = ROOT / config["output_dir"] / "best_model.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.to(device)

        test_dir = ROOT / config["test_soundscapes_dir"]
        test_files = sorted(test_dir.glob("*.ogg"))
        output_path = ROOT / config["submission_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generate_submission(model, [str(f) for f in test_files], species_list, config, device, output_path)


if __name__ == "__main__":
    main()

from multiprocessing.sharedctypes import RawValue
from random import random
import tempfile
import subprocess
import lib
import os
import optuna
import argparse
from pathlib import Path
from train_sample_ctabganp import train_ctabgan, sample_ctabgan
from scripts.eval_catboost import train_catboost

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()
real_data_path = args.data_path
eval_type = args.eval_type
train_size = args.train_size
device = args.device
assert eval_type in ('merged', 'synthetic')

def objective(trial):
    
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    
    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 6, 8
    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    ####

    steps = trial.suggest_categorical('steps', [1000, 5000, 10000])
    # steps = trial.suggest_categorical('steps', [10])
    batch_size = 2 ** trial.suggest_int('batch_size', 9, 11)
    random_dim = 2 ** trial.suggest_int('random_dim', 4, 7)
    num_channels = 2 ** trial.suggest_int('num_channels', 4, 6)

    # steps = trial.suggest_categorical('steps', [1000])

    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -2, 3)))

    train_params = {
        "lr": lr,
        "epochs": steps,
        "class_dim": d_layers,
        "batch_size": batch_size,
        "random_dim": random_dim,
        "num_channels": num_channels
    }
    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        ctabgan = train_ctabgan(
            parent_dir=dir_,
            real_data_path=real_data_path,
            train_params=train_params,
            change_val=True,
            device=device
        )

        for sample_seed in range(5):
            sample_ctabgan(
                ctabgan,
                parent_dir=dir_,
                real_data_path=real_data_path,
                num_samples=num_samples,
                train_params=train_params,
                change_val=True,
                seed=sample_seed,
                device=device
            )

            T_dict = {
                "seed": 0,
                "normalization": None,
                "num_nan_policy": None,
                "cat_nan_policy": None,
                "cat_min_frequency": None,
                "cat_encoding": None,
                "y_policy": "default"
            }
            metrics = train_catboost(
                parent_dir=dir_,
                real_data_path=real_data_path, 
                eval_type=eval_type,
                T_dict=T_dict,
                change_val=True,
                seed = 0
            )

            score += metrics.get_val_score()
    return score / 5


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=35, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/ctabgan-plus/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/ctabgan-plus/",
    "real_data_path": real_data_path,
    "seed": 0,
    "device": args.device,
    "train_params": study.best_trial.user_attrs["train_params"],
    "sample": {"seed": 0, "num_samples": study.best_trial.user_attrs["num_samples"]},
    "eval": {
        "type": {"eval_model": "catboost", "eval_type": eval_type},
        "T": {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        },
    }
}

train_ctabgan(
    parent_dir=f"exp/{Path(real_data_path).name}/ctabgan-plus/",
    real_data_path=real_data_path,
    train_params=study.best_trial.user_attrs["train_params"],
    change_val=False,
    device=device
)

lib.dump_config(config, config["parent_dir"]+"config.toml")

subprocess.run(['python3.9', "scripts/eval_seeds.py", '--config', f'{config["parent_dir"]+"config.toml"}',
                '10', "ctabgan-plus", eval_type, "catboost", "5"], check=True)
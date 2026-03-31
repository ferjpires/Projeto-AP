import random

import pandas as pd

try:
    import optuna
except ImportError:
    optuna = None

def build_random_search(dictionary, n_iter=20, random_state=42):
    random.seed(random_state)
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    for _ in range(n_iter):
        instance = [random.choice(v) for v in values]
        yield dict(zip(keys, instance))


def suggest_params(trial, search_space):
    params = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict):
            param_type = spec.get("type", "categorical")
            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                    log=spec.get("log", False),
                )
            elif param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step"),
                    log=spec.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unsupported search space type for '{name}': {param_type}")
        else:
            params[name] = trial.suggest_categorical(name, spec)
    return params


def create_study(direction="maximize", study_name=None, seed=42):
    if optuna is None:
        raise ImportError("optuna is not installed")

    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=sampler,
    )


def study_results_dataframe(study):
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            "state": trial.state.name,
        }
        row.update(trial.params)
        row.update(trial.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)

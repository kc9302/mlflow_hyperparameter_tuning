import logging
import optuna
from hyperparameter_tuning import objective
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger().setLevel(logging.DEBUG)


def run():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                direction="maximize",
                                pruner=optuna.pruners.MedianPruner())

    study.optimize(objective, n_trials=10)
    logging.debug(study.best_params)


if __name__ == "__main__":
    run()

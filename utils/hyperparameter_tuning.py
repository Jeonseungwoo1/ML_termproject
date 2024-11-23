import time
import argparse
import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from collections import defaultdict
from surprise.dataset import DatasetAutoFolds
from surprise import SVD, NormalPredictor
from surprise.model_selection import GridSearchCV
from experiment import load_config


''' 

Parameters:
    epochs (int): The number of epochs for training the model on the training data.
    lr_all (float): The learning rate used for all parameters during training.
    n_factors (int): The number of latent factors in the latent factor space.

'''
def hyperparm_tuning(dataset, param_grid):
    start = time.time()
    time.sleep(1)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(dataset[['user_id', 'recipe_id', 'rating']], reader)

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    best_rmse = gs.best_score['rmse']
    epochs = gs.best_params['rmse']['n_epochs']
    lr = gs.best_params['rmse']['lr_all']
    factors = gs.best_params['rmse']['n_factors']

    return best_rmse, epochs, lr, factors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommend System Script")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    np.random.seed(42)
    param_grid = config["param_grid"]
    print("To tune the hyperparameters, GridSearchCV was used. The following parameter candidates were explored:")
    for param, values in param_grid.items():
        print(f"- {param}: {values}")
    rating_data = pd.read_csv(config["svd_datasets"]["validation"])


    print("\n#### Hyperparameters tuning is Start! #### --- {}".format(time.strftime('%m.%d_%Hh%Mm%Ss')))
    best_rmse, epochs, lr, factors = hyperparm_tuning(rating_data, param_grid)
    print("#### Hyperparameters tuning finished! #### --- {}".format(time.strftime('%m.%d_%Hh%Mm%Ss')))
    print("\n#### Best parameters ####")
    print("Best RMSE: {}".format(best_rmse))
    print("Epochs: {}, lr: {}, factors: {}".format(epochs, lr, factors))
import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
from surprise.dataset import DatasetAutoFolds
from surprise import SVD, NormalPredictor
from surprise.model_selection import GridSearchCV

import time

def hyperparm_tunig(path, dataset, param_grid):
    start = time.time()
    time.sleep(1)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(dataset[['user_id', 'recipe_id', 'rating']], reader)

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mse'], cv=3)
    gs.fit(data)

    epochs = gs.best_params['rmse']['n_epochs']
    lr = gs.best_params['rmse']['lr_all']
    factors = gs.best_params['rmse']['n_factors']

    print(f"수행 시간: {time.time() - start:.4f} sec")

    return epochs, lr, factors
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise.dataset import DatasetAutoFolds
from surprise import SVD, NormalPredictor
from surprise.model_selection import GridSearchCV
from hyperparameter_tuning import hyperparm_tunig

def dataset_load(path1, path2):
    recipe_data = pd.read_csv(path1)
    rating_data = pd.read_csv(path2)
    return recipe_data, rating_data

def train(path2, model):
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5))
    data_folds = DatasetAutoFolds(ratings_file=path2, reader=reader)
    trainset = data_folds.build_full_trainset()

    model.fit(trainset)

def get_non_rated_recipe(recipe_data, rating_data, user_id):
    rated_recipes = rating_data[rating_data['user_id'] == user_id]['recipe_id'].tolist()

    total_recipes = recipe_data['id'].tolist()

    non_rated_recipes = [recipe for recipe in total_recipes if recipe not in rated_recipes]
    print('평점 매긴 레시피 수:',len(rated_recipes), '추천대상 레시피 수:', len(non_rated_recipes), '전체 레시피 수:', len(total_recipes))

    return non_rated_recipes

def recomm_recipe_by_surprise(algo, user_id, non_rated_recipes, top_n=10):
    predictions = [algo.predict(str(user_id), str(recipe_id)) for recipe_id in non_rated_recipes]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    top_recipe_ids = [int(pred.iid) for pred in top_predictions]
    top_recipe_rating = [pred.est for pred in top_predictions]
    top_recipe_name = recipe_data[recipe_data.id.isin(top_recipe_ids)]['name']
    top_recipe_preds = [(id, name, rating) for id, name, rating in zip(top_recipe_ids, top_recipe_name, top_recipe_rating)]

    return top_recipe_preds


if __name__ == '__main__':

    path1 = "./dataset/RAW_recipes.csv"
    path2 = "./dataset/PP_interactions_train.csv"
    path3 = "./dataset/Remove_Header_interaction_train.csv"
    
    param_grid = {
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.010],
        'n_factors': [50, 100]
    }
    recipe_data, rating_data = dataset_load(path1, path2)
    #epochs, lr, factors = hyperparm_tunig(path2, rating_data, param_grid)
    epochs, lr, factors= 20, 0.005, 50
    model = SVD(n_epochs=epochs, lr_all=lr, n_factors=factors, random_state=0)
    train(path3, model)

    non_rated_recipes = get_non_rated_recipe(recipe_data, rating_data, 2695)
    top_recipe_preds = recomm_recipe_by_surprise(model, 2695, non_rated_recipes, top_n=10)

    print('##### Top-10 추천 레시피 리스트 #####')

    for top_recipe in top_recipe_preds:
        print(top_recipe[1], ":", top_recipe[2])

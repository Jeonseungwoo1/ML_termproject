import argparse
import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise.dataset import DatasetAutoFolds
from surprise import SVD, NormalPredictor
from surprise.model_selection import GridSearchCV
from utils.experiment import load_config

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
    parser = argparse.ArgumentParser(description="Recommend System Script")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    
    recipe_data, rating_data = dataset_load(config["datasets"]["recipe"], config["datasets"]["review"])
   
    model = SVD(n_epochs=config["parameter"]["epochs"],
                lr_all=config["parameter"]["lr"],
                n_factors=config["parameter"]["factors"],
                random_state=config["parameter"]["random_state"])
    
    train(config["datasets"]["remove_header_review"], model)

    if config["target"]["user_id"] in rating_data['user_id'].values:
        non_rated_recipes = get_non_rated_recipe(recipe_data, rating_data, config["target"]["user_id"])
        top_recipe_preds = recomm_recipe_by_surprise(model, config["target"]["user_id"], non_rated_recipes, config["target"]["top_n"])

        print('#####Top-10 recommended recipe lists (for user: {}) #####'.format(config["target"]["user_id"]))
        for top_recipe in top_recipe_preds:
            print(top_recipe[1], ":", round(top_recipe[2],2))
    else:
        raise ValueError(f'User ID {config["target"]["user_id"]} does not exist in the file. Program will terminate.')


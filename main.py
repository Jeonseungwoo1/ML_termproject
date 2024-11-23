import argparse
import numpy as np
import pandas as pd

from utils.experiment import load_config
from utils.data_loader import load_data
from recommender.recommender import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommend System Script")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    np.random.seed(42)
    setting = config["setting"]["algo"]

    if setting == "SVD":
        recommender = SVD_Recommender(config)
        recommender.display_recommendations()
    elif setting == "User_based":
        recommender = UserBasedRecommender(config)
        recommender.display_recommender()
    elif setting == "Item_based":
        recommender = ItemBasedRecommender(config)
        recommender.display_recommendations()
    elif setting == "Content_based":
        recommender = ContentBasedRecommender(config)
        recommender.display_recommendations()
    elif setting == "TruncatedSVD":
        recommender = TruncatedSVD_Recommender(config)
        recommender.run()
    else:
        print("Please check 'setting' in config.json file")
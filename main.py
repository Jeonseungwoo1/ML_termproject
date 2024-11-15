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

    setting = "SVD"
    if setting == "SVD":
        recommender = SVD_Recommender(config)
        recommender.display_recommendations()
    elif setting == "User_based":
        recommender = UserBasedRecommender(config)
        data = load_data_for_userbased(config["user_based"]["dataset"])
        train_data, test_data = train_test_split(data, test_size=0.2)
        recommender.display_recommender(train_data, test_data)
    elif setting == "Item_based":
        recommender = ItemBasedRecommender(config)
        recommender.display_recommendations()
    else:
        recommender = ContentBasedRecommender(config)
        recommender.display_recommendations()
        


        
        
    

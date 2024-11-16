import pandas as pd
from  surprise import Dataset, Reader

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_data_for_contentbased(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['user_id', 'recipe_id', 'rating'])

def load_data_for_userbased(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['user_id', 'recipe_id', 'rating'])
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

    return data, df

def load_data_for_movie(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    return data, df
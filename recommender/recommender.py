import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from utils.data_loader import load_data, load_data_for_userbased, load_data_for_movie
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


from surprise import SVD, Dataset, Reader, KNNBasic, accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split



class ItemBasedRecommender:
    def __init__(self,config):
        self.config = config
        self.interactions_df = load_data(self.config["item_based_datasets"]["interactions"])
        self.recipes_df = load_data(self.config["item_based_datasets"]["recipes"])
        self.test_df = load_data(self.config["item_based_datasets"]["test"])
    def calculate_similarity(self, ratings_matrix):
        item_similarity = cosine_similarity(ratings_matrix.T)
        return pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)
    def get_recommendations(self, user_id, df):
        ratings_matrix = df.pivot_table(index='user_id', columns='recipe_id', values='rating', fill_value=0)
        item_similarity_df = self.calculate_similarity(ratings_matrix)
        
        if user_id in ratings_matrix.index:
            user_ratings = ratings_matrix.loc[user_id]
            similar_items = pd.Series(dtype=float)

            for item, rating in user_ratings.items():
                if rating > 0:
                    similar_items = pd.concat([similar_items, item_similarity_df[item] * rating])

            similar_items = similar_items.sort_values(ascending=False)
            similar_items = similar_items[~similar_items.index.isin(user_ratings[user_ratings > 0].index)]
            return similar_items.head(self.config["item_based"]["num_recommendations"])
        else:
            return pd.Series(dtype=float)
    
    def sample_recommendations(self):
        all_similar_items = pd.Series(dtype=float)
        sampled_user_id = None

        for i in range(self.config["item_based"]["num_sample"]):
            print(f"Sampling... {i + 1}/{self.config["item_based"]["num_sample"]}")
            sample_df = self.interactions_df.sample(n=self.config["item_based"]["sample_size"], replace=False)

            if i == 0:
                sampled_user_id = sample_df.iloc[0, 0]

            similar_items = self.get_recommendations(sampled_user_id, sample_df)
            all_similar_items = pd.concat([all_similar_items, similar_items])

        all_similar_items = all_similar_items.sort_values(ascending=False).head(self.config["item_based"]["num_recommendations"])
        return sampled_user_id, all_similar_items

    def display_recommendations(self):
        user_id, recommendations = self.sample_recommendations()
        print(f"Recommendations for User ID: {user_id}")
        for recipe_id in recommendations.index:
            recipe_name = self.recipes_df.loc[self.recipes_df['id'] == recipe_id, 'name'].tolist()
            if recipe_name:
                print(f"- {recipe_name[0]}")
        print("End of Recommendations")

        #self.evaluate(user_id, recommendations)
        
class SVD_Recommender:
    def __init__(self, config):
        self.config = config
        self.user_id = config["target"]["user_id"]
        self.recipe_df = load_data(config["svd_datasets"]["recipe"])
        self.rating_df = load_data(config["svd_datasets"]["review"])
        self.movie_ratings_df = load_data(config["movie"]["ratings"])
        self.movies_df = load_data(config["movie"]["movies"])
        self.test_df = pd.read_csv(self.config["svd_datasets"]["test"]).dropna(subset=['user_id', 'recipe_id', 'rating'])
        self.validation_df = pd.read_csv(self.config["svd_datasets"]["validation"]).dropna(subset=['user_id', 'recipe_id', 'rating'])

    def train(self, model):
        reader = Reader(rating_scale=(0,5))
        if self.config["movie"]["use"] == "False":
            data = Dataset.load_from_df(self.rating_df[['user_id', 'recipe_id', 'rating']], reader)
        else:
            data = Dataset.load_from_df(self.movie_ratings_df[['userId', 'movieId', 'rating']], reader) 
        trainset = data.build_full_trainset()

        model = model.fit(trainset)
        return model

    def evaluate(self, model):
        print("\nModel Evaluation Start ...")
        threshold = self.config["evaluate"]["threshold"]
        testset = list(self.test_df[['user_id', 'recipe_id', 'rating']].itertuples(index=False, name=None))
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False) 
        y_true = []
        y_pred = []

        for pred in predictions:
            true_rating = pred.r_ui
            estimated_rating = pred.est

            y_true.append(int(true_rating >= threshold)) 
            y_pred.append(int(estimated_rating >= threshold))

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        #Coverage
        total_items = len(self.rating_df['recipe_id'].unique())
        recommended_items = {int(pred.iid) for pred in predictions if pred.est >= threshold}
        coverage = len(recommended_items)/ total_items

        #Diversity
        item_index = np.array(list(recommended_items))
        if item_index.size > 0:
            if item_index.ndim == 1:
                item_index = item_index.reshape(-1,1)
            item_vectors = cosine_similarity(item_index)
            np.fill_diagonal(item_vectors, 0)
            avg_similarity = np.mean(item_vectors)
            diversity = 1 - avg_similarity
        else:
            diversity = 0

        # Novelty
        item_popularity = self.rating_df['recipe_id'].value_counts().to_dict()
        novelty = np.mean([np.log2(1+ item_popularity.get(int(pred.iid), 1)) for pred in predictions])

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Coverage': coverage,
            'Diversity': diversity,
            'Novelty': novelty
        } 
        
        print("######### Evaluation Metrics #########")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        self.visualize_metrics(metrics)
    
    def visualize_metrics(self, metrics):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = [
        '#1f77b4',  # RMSE (파란색)
        '#ff7f0e',  # MAE (주황색)
        '#2ca02c',  # Precision (녹색)
        '#d62728',  # Recall (빨간색)
        '#9467bd',  # F1-score (보라색)
        '#8c564b',  # Coverage (갈색)
        '#e377c2',  # Diversity (분홍색)
        '#7f7f7f',  # Novelty (회색)
        ]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(metric_names, metric_values, color=colors[:len(metric_names)], edgecolor='black')


        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+2, f"{value:.4f}", ha='center', va='bottom', fontsize=10)

        plt.title("SVD Recommendation System Performance Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.ylim(0, max(metric_values) * 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def recommend_popular_recipe(self):
            popular_recipes = self.rating_df['recipes_id'].value_counts().head(self.config["user_based"]["top_n"]).index
            recommendations = self.recipe_df[self.recipe_df['id'].isin(popular_recipes)]
            print("\nTop 10 Popular Recipes:")
            for i, (recipe_id, name) in enumerate(zip(recommendations['id'], recommendations['name']), 1):
                print(f"{i}. Recipe ID: {recipe_id}, Name: {name}")  
    def random_user(self):
        if self.config["movie"]["use"] == "False":
            user_ids  = self.test_df['user_id'].unique()
            random_user_id = np.random.choice(user_ids)
        else:
            user_ids = self.movie_ratings_df['userId'].unique()
            random_user_id = np.random.choice(user_ids)
        return random_user_id

    def get_non_rated_recipe(self, user_id):
        if self.config["movie"]["use"] == "False":
            rated_recipes = self.rating_df[self.rating_df['user_id'] == user_id]['recipe_id'].tolist()
            total_recipes = self.recipe_df['id'].tolist()
        else:
            rated_recipes = self.movie_ratings_df[self.movie_ratings_df['userId'] == user_id]['movieId'].tolist()
            total_recipes = self.movies_df['movieId'].tolist()

        non_rated_recipes = [recipe for recipe in total_recipes if recipe not in rated_recipes]
        
        if self.config["movie"]["use"] == "False":
            print('|','평점 매긴 레시피 수:',len(rated_recipes),'|', '추천대상 레시피 수:', len(non_rated_recipes), '|', '전체 레시피 수:', len(total_recipes),'|')
        else:
            print('|','평점 매긴 영화 수:',len(rated_recipes),'|', '추천대상 영화 수:', len(non_rated_recipes), '|', '전체 영화 수:', len(total_recipes),'|')
        return non_rated_recipes
    
    def recomm_recipe_by_surprise(self):
        model = SVD(n_epochs=self.config["parameter"]["epochs"],
                lr_all=self.config["parameter"]["lr"],
                n_factors=self.config["parameter"]["factors"],
                random_state=42
                )

        print("SVD Recommender model Train Start...")
        model = self.train(model)
        print("SVD Recommender model Train End...!\n")

        random_user_id = self.random_user()
        if self.config["movie"]["use"] == "False":
            print("Start Recommender recipe for User {}...".format(random_user_id))
        else:
            print("Start Recommender movie for User {}...".format(random_user_id))
        
        non_rated_recipes = self.get_non_rated_recipe(random_user_id)


        if self.config["movie"]["use"] == "False":
            predictions = [model.predict(str(random_user_id), str(recipe_id)) for recipe_id in non_rated_recipes]
        else:
            predictions = [model.predict(str(random_user_id), str(movieId)) for movieId in non_rated_recipes]
        predictions.sort(key=lambda x: x.est, reverse=True)


        top_predictions = predictions[:self.config["target"]["top_n"]]
        top_recipe_ids = [int(pred.iid) for pred in top_predictions]
        top_recipe_rating = [pred.est for pred in top_predictions]

        if self.config["movie"]["use"] == "False":
            top_recipe_name = self.recipe_df.set_index('id').loc[top_recipe_ids]['name'].tolist()
        else:
            top_recipe_name = self.movies_df.set_index('movieId').loc[top_recipe_ids]['title'].tolist()
        top_recipe_preds = [(id, name, rating) for id, name, rating in zip(top_recipe_ids, top_recipe_name, top_recipe_rating)]

        return model,top_recipe_preds, random_user_id

    def display_recommendations(self):
        
        if self.config["movie"]["use"] == "False":
            model, top_recipe_preds, random_user_id= self.recomm_recipe_by_surprise()
            print('#####Top-10 recommended recipe lists (for user: {}) #####'.format(random_user_id))
            for top_recipe in top_recipe_preds:
                print(top_recipe[1], ":", round(top_recipe[2],4))
            self.evaluate(model) 
        else:
            model, top_movie_preds, random_user_id, test= self.recomm_recipe_by_surprise()
            print('#####Top-10 recommended movie lists (for user: {}) #####'.format(random_user_id))
            for top_movie in top_movie_preds:
                print(top_movie[1], ":", round(top_movie[2],4))
  
class ContentBasedRecommender:
    def __init__(self, config):
        self.config = config
        self.train = load_data(config["content_based"]["train"])
        self.test = load_data(config["content_based"]["test"])
       
    def vectorize(self, data):
        data['ingredients'] = data['ingredients'].apply(lambda x: x.lower().strip('"').strip("'"))
        data['name'] = data["name"].apply(lambda x: x.lower() if isinstance(x, str) else x)
        data['combined_features'] = data['name'] + data['ingredients']

        data["combined_features"] = data['combined_features'].fillna('')

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data['combined_features'])

        return data, tfidf_matrix
    
    def suggetest_popular_recipes(self):
        popular_recipes = self.train['id'].value_counts().head(self.config["content_based"]["top_n"]).index
        recommendations = self.train[self.train['id'].isin(popular_recipes)]
        print("#### Top 10 Popular Recipes: ####")
        for rank, (index, row) in enumerate(recommendations.iterrows(), start=1):
            print(f"Recipe Ranking: {rank}")
            print(f"Recipe Name: {row['name'].title()}\n")
            print(f"Ingredients: {self.clean_ingredients(row['ingredients'])}\n")
            print(f"Cooking Time: {row['minutes']} minutes\n")
            print("-" * 80 + "\n")

    def suggest_recipes_based_on_ratings(self):
        config = self.config
        dataset = self.train
        data, tfidf_matrix= self.vectorize(dataset)
        user_ids  = dataset['user_id'].unique()
        random_user_id = np.random.choice(user_ids)

        ann_model = NearestNeighbors(n_neighbors=config["content_based"]["n_neighbors"],
                                     algorithm=config["content_based"]["algo"],
                                     metric=config["content_based"]["metric"])
        ann_model.fit(tfidf_matrix)
        user_ratings = data[data['user_id'] == random_user_id]
        high_rated_recipes = user_ratings[user_ratings['rating'] >= config["content_based"]["min_rating"]]

        if not high_rated_recipes.empty:
            top_recipe_id = high_rated_recipes.iloc[0]['id']
            recipe_idx = data.index[data['id'] == top_recipe_id].tolist()[0]

            _, recipe_indices = ann_model.kneighbors(tfidf_matrix[recipe_idx], n_neighbors=config["content_based"]["top_n"]+1)
            recipe_indices = recipe_indices[0][1:]

            return data, tfidf_matrix, random_user_id, data.iloc[recipe_indices], ann_model
        else:
            print(f"User {random_user_id} has no high-rated recipes with a rating >= {config["content_based"]["min_rating"]}.")
            print("Showing popular recipes instead.")
            self.suggetest_popular_recipes()
            print("Exiting the Program...")
            exit() 
            
    
    def clean_ingredients(self, ingredients):
        cleaned_ingredients = ingredients.strip("[]").replace("'", "").replace('"', '').split(', ')
        cleaned_ingredients = [ingredient.capitalize() for ingredient in cleaned_ingredients]
        return ', '.join(cleaned_ingredients)
            

    def display_recommendations(self):
        data,tfidf_matrix, target_id, recipes, model= self.suggest_recipes_based_on_ratings()

        if not recipes.empty:
            print("### Content based Recommendations for {} ###".format(target_id))
            for rank, (index, row) in enumerate(recipes.iterrows(), start=1):
                print(f"Recipe Ranking: {rank}")
                print(f"Recipe Name: {row['name'].title()}\n")
                print(f"Ingredients: {self.clean_ingredients(row['ingredients'])}\n")
                print(f"Cooking Time: {row['minutes']} minutes\n")
                print("-" * 80 + "\n")
        else:
            print("No recommendations available.")

        #self.evaluate(data, tfidf_matrix, model)
        
class UserBasedRecommender:
    def __init__(self, config):
        self.config = config
        self.data, self.ratings_df = load_data_for_userbased(config["user_based"]["rating_train"])
        self.recipes_df = load_data(config["user_based"]["recipes"])
        self.movie_data, self.movie_ratings_df = load_data_for_movie(config["movie"]["ratings"])
        self.movies_df = load_data(config["movie"]["movies"])
        self.k = self.config["user_based"]["k"]
        self.similarity=self.config["user_based"]["similarity"]
    def user_based_cf_model(self):
        if self.config["movie"]["use"] == "False":
            trainset = self.data.build_full_trainset()
        else:
            trainset = self.movie_data.build_full_trainset()
        sim_options = {
            'name' : self.similarity,
            'user_based': True
        }

        knn_model = KNNBasic(k=self.k, sim_options=sim_options)
        knn_model.fit(trainset)

        print(f"User-Based CF Model trained with k={self.k}, similarity='{self.similarity}'")
        return knn_model, trainset

    def recommend_recipes(self, model, user_id, trainset):
        try:
            inner_user_id = trainset.to_inner_uid(user_id)
        except ValueError:
            print("User ID not found in training data.")
            return False
        
        all_recipe_ids = set(trainset.all_items())
        rated_recipe_ids = {
            trainset.to_inner_iid(iid)
            for iid in trainset.ur[inner_user_id]
            if iid in trainset._raw2inner_id_items}
        unrated_recipe_ids = all_recipe_ids - rated_recipe_ids

        predictions = [model.predict(user_id, trainset.to_raw_iid(iid)) for iid in unrated_recipe_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)

        top_n_recommendations = predictions[:self.config["user_based"]["top_n"]]
        
        if self.config["movie"]["use"] == "False":
            print(f"\nTop 10 Recommended Recipes for User({user_id}):")
            header = f"| {'Recipe ID'.ljust(10)} | {'Name'.ljust(60)} | {'Predicted Rating'.ljust(15)}"
            print(header)
            print("-" * len(header))
            for i, pred in enumerate(top_n_recommendations, 1):
                recipe_id = int(pred.iid)
                recipe_info = self.recipes_df[self.recipes_df['id'] == recipe_id]
                if not recipe_info.empty:
                    name = recipe_info['name'].values[0]
                    predicted_rating = round(pred.est, 4)

                    # 출력 형식 조정
                    print(f"| {str(recipe_id).ljust(10)} | {name.ljust(60)} | {str(predicted_rating).ljust(15)}")
            return True
        else:
            print(f"\nTop 10 Recommended Movies for User({user_id})")
            header = f"| {'Movie ID'.ljust(10)} | {'Title'.ljust(60)} | {'Genres'.ljust(40)} |"
            print(header)
            print("-" * len(header))
            for i, pred in enumerate(top_n_recommendations, 1):
                movieId = int(pred.iid)
                movies_info = self.movies_df[self.movies_df['movieId'] == movieId]
                if not movies_info.empty:
                    title = movies_info['title'].values[0]
                    genres = movies_info['genres'].values[0]

                    # 출력 형식 조정
                    print(f"| {str(movieId).ljust(10)} | {title.ljust(60)} | {genres.ljust(40)} |")
            return True

        

    def recommend_popular_recipe(self):
        popular_recipes = self.ratings_df['recipe_id'].value_counts().head(self.config["user_based"]["top_n"]).index
        recommendations = self.recipes_df[self.recipes_df['id'].isin(popular_recipes)]
        print("\nTop 10 Popular Recipes:")
        for i, (recipe_id, name, steps) in enumerate(zip(recommendations['id'], recommendations['name'], recommendations['steps']), 1):
            print(f"{i}. Recipe ID: {recipe_id}, Name: {name}, Steps: {steps}")
    def evaluate(self, model, trainset):
        test_df = pd.read_csv(self.config["user_based"]["rating_validation"]).dropna(subset=['user_id', 'recipe_id', 'rating'])
        testset = list(test_df[['user_id', 'recipe_id', 'rating']].itertuples(index=False, name=None))

        print("\nModel Evaluation Start ...")
        predictions = model.test(testset)
        
        if not predictions:
            print("Prediction list is empty. Skipping evaluation.")
            return
        #RMSE, MAE
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        threshold = self.config["evaluate"]["threshold"]
        
        #Precision, Recall, F1-score
        y_true = [int(pred.r_ui >= threshold) for pred in predictions]
        y_pred = [int(pred.est >= threshold) for pred in predictions]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        #Coverage
        total_items = len(self.ratings_df['recipe_id'].unique())
        recommended_items = {int(pred.iid) for pred in predictions if pred.est >= threshold}
        coverage = len(recommended_items)/ total_items

        #Diversity
        item_index = np.array(list(recommended_items))
        if item_index.size > 1:
            if item_index.ndim == 1:
                item_index = item_index.reshape(-1,1)
            item_vectors = cosine_similarity(item_index)
            np.fill_diagonal(item_vectors, 0)
            avg_similarity = np.mean(item_vectors)
            diversity = 1 - avg_similarity
        else:
            diversity = 0

        #Novelty
        item_popularity = self.ratings_df['recipe_id'].value_counts().to_dict()
        novelty = np.mean([np.log2(1+ item_popularity.get(int(pred.iid), 1)) for pred in predictions])

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Coverage': coverage,
            'Diversity': diversity,
            'Novelty': novelty
        } 
        
        print("######### Evaluation Metrics #########")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        self.visualize_metrics(metrics)

    def visualize_metrics(self, metrics):
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = [
        '#1f77b4',  # RMSE (파란색)
        '#ff7f0e',  # MAE (주황색)
        '#2ca02c',  # Precision (녹색)
        '#d62728',  # Recall (빨간색)
        '#9467bd',  # F1-score (보라색)
        '#8c564b',  # Coverage (갈색)
        '#e377c2',  # Diversity (분홍색)
        '#7f7f7f',  # Novelty (회색)
        ]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(metric_names, metric_values, color=colors[:len(metric_names)], edgecolor='black')


        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()+2, f"{value:.4f}", ha='center', va='bottom', fontsize=10)

        plt.title("User Based Recommendation System Performance Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.ylim(0, max(metric_values) * 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def display_recommender(self):
        
        user_based_model, trainset = self.user_based_cf_model()
        if self.config["movie"]["use"] == "False":
            user_ids  = self.ratings_df['user_id'].unique()
            random_user_id = np.random.choice(user_ids)
        else:
            user_ids  = self.movie_ratings_df['userId'].unique()
            random_user_id = np.random.choice(user_ids)

        user_exists = self.recommend_recipes(user_based_model, random_user_id, trainset)

        if not user_exists:
            print("User ID not found. Showing popular recipes instead.")
            self.recommend_popular_recipe()

        if self.config["movie"]["use"] == "False":
            self.evaluate(user_based_model, trainset)

class TruncatedSVD_Recommender:
    def __init__(self, config):
        self.config = config
        self.user_id = config["target"]["user_id"]
        self.recipe_df = load_data(config["svd_datasets"]["recipe"])
        self.rating_df = load_data(config["svd_datasets"]["review"])
        self.movie_ratings_df = load_data(config["movie"]["ratings"])
        self.movies_df = load_data(config["movie"]["movies"])
        self.n_models = 2
        self.models = []
        self.user_index = None
        self.item_index = None

    
    def train(self, model, user_recipe_matrix):
        print("Training Truncated SVD model...")
        user_features = model.fit_transform(user_recipe_matrix)
        recipe_features = model.components_.T
        print("Training completed.")

        return user_features, recipe_features



    def predict_ratings(self, user_features, recipe_features):
        predicted_ratings = np.dot(user_features, recipe_features.T)
        return predicted_ratings


    

    def evaluate(self, user_recipe_matrix, predicted_ratings):
        if isinstance(user_recipe_matrix, np.ndarray):
            true_ratings = user_recipe_matrix.flatten()
        else:
            true_ratings = user_recipe_matrix.toarray().flatten()

        predicted_ratings = predicted_ratings.flatten()
        mask = true_ratings != 0

        rmse = np.sqrt(mean_squared_error(true_ratings[mask], predicted_ratings[mask]))
        mae = mean_absolute_error(true_ratings[mask], predicted_ratings[mask])

        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")



    def recommend(self, user_id, user_recipe_matrix, predicted_ratings, user_index, item_index, top_n=10):
        try:
            user_idx = np.where(user_index == user_id)[0][0]
        except IndexError:
            print(f"User ID {user_id} not found in the index.")
            return

        if isinstance(user_recipe_matrix, np.ndarray):
            user_ratings = user_recipe_matrix[user_idx].flatten()
        else:
            user_ratings = user_recipe_matrix[user_idx].toarray().flatten()

        unrated_indices = np.where(user_ratings == 0)[0]
        recommendations = predicted_ratings[user_idx, unrated_indices]

        top_indices = unrated_indices[np.argsort(recommendations)[-top_n:]]
        recommended_ids = item_index[top_indices]

        if self.config["movie"]["use"] == "False":
            recommended_items = self.recipe_df[self.recipe_df['id'].isin(recommended_ids)]
            print("\nTop-10 recommended recipes with predicted ratings:")
        else:
            recommended_items = self.movies_df[self.movies_df['movieId'].isin(recommended_ids)]
            print("\nTop-10 recommended movies with predicted ratings:")

        max_id_length = max(len(str(item_id)) for item_id in recommended_ids)
        max_name_length = max(len(str(name)) for name in recommended_items['name'])

        max_id_length = max(max_id_length, len("Item ID"))
        max_name_length = max(max_name_length, len("Name"))

        header = f"{' Item ID'.ljust(max_id_length)} | {'Name'.ljust(max_name_length)}"
        print(header)
        print("-" * len(header))

        for i, (item_id, name) in enumerate(zip(recommended_ids, recommended_items['name']), 1):
            print(f" {str(item_id).ljust(max_id_length)} | {name.ljust(max_name_length)}")


    def run(self):
        if self.config["movie"]["use"] == "False":
            sampled_data = self.rating_df.sample(frac=0.3, random_state=42)
            user_recipe_matrix_df = sampled_data.pivot(index='user_id', columns='recipe_id', values='rating').fillna(0)
        else:
            sampled_data = self.movie_ratings_df.sample(frac=0.3, random_state=42)
            user_recipe_matrix_df = sampled_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        user_index = user_recipe_matrix_df.index
        item_index = user_recipe_matrix_df.columns
        user_recipe_matrix = user_recipe_matrix_df.values

        n_components = min(self.config["parameter"]["factors"], user_recipe_matrix.shape[1] // 2)
        model = TruncatedSVD(n_components=n_components, random_state=42)
        user_features, recipe_features = self.train(model, user_recipe_matrix)

        predicted_ratings = self.predict_ratings(user_features, recipe_features)
        self.evaluate(user_recipe_matrix, predicted_ratings)

        print("Recommend Start...")

        random_user_id = np.random.choice(user_index)
        print(f"Selected random user ID: {random_user_id}")

        self.recommend(random_user_id, user_recipe_matrix, predicted_ratings, user_index, item_index)

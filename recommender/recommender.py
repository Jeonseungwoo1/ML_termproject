import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from utils.data_loader import load_data, load_data_for_userbased
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.sparse import csr_matrix
from surprise import SVD, Dataset, Reader, KNNBasic, accuracy
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split



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
    def evaluate(self, sampled_user_id, recommendations):
        # 평가를 위한 카운터 초기화
        hits = 0
        total_relevant = 0
        total_recommended = 0

        # 학습 데이터를 샘플링
        sample_df = self.interactions_df.sample(n=self.config["item_based"]["sample_size"], replace=False)
        print(f"샘플링된 학습 데이터 크기: {sample_df.shape}")

        # 샘플링된 사용자에 대해서만 평가 진행
        test_user_data = self.test_df[self.test_df['user_id'] == sampled_user_id]
        true_items = set(test_user_data['recipe_id'].tolist())
    

        if recommendations.empty:
            print("추천 목록이 비어 있습니다. 평가를 건너뜁니다.")
            return 0, 0, 0

        recommended_items = set(recommendations.index)

        # Hit 수 계산
        hits += len(recommended_items & true_items)
        total_relevant += len(true_items)
        total_recommended += len(recommended_items)

        # Precision, Recall, F1-Score 계산
        precision = hits / total_recommended if total_recommended > 0 else 0
        recall = hits / total_relevant if total_relevant > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"평가 결과 (샘플링된 사용자 ID: {sampled_user_id}):")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

        return precision, recall, f1_score

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
        self.test_df = pd.read_csv(self.config["svd_datasets"]["test"]).dropna(subset=['user_id', 'recipe_id', 'rating'])

    def train(self, model, trainset):
        reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5))
        data_folds = DatasetAutoFolds(ratings_file=self.config["svd_datasets"]["remove_header_review"], reader=reader)
        trainset = data_folds.build_full_trainset()

        model.fit(trainset)

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
        user_ids  = self.test_df['user_id'].unique()
        random_user_id = np.random.choice(user_ids)
        return random_user_id

    def get_non_rated_recipe(self, user_id):
        rated_recipes = self.rating_df[self.rating_df['user_id'] == user_id]['recipe_id'].tolist()
        total_recipes = self.recipe_df['id'].tolist()

        non_rated_recipes = [recipe for recipe in total_recipes if recipe not in rated_recipes]
        print('|','평점 매긴 레시피 수:',len(rated_recipes),'|', '추천대상 레시피 수:', len(non_rated_recipes), '|', '전체 레시피 수:', len(total_recipes),'|')

        return non_rated_recipes
    
    def recomm_recipe_by_surprise(self):
        model = SVD(n_epochs=self.config["parameter"]["epochs"],
                lr_all=self.config["parameter"]["lr"],
                n_factors=self.config["parameter"]["factors"],
                random_state=self.config["parameter"]["random_state"])
        reader =Reader(line_format='user item rating', sep=',', rating_scale=(0, 5))
        train_data = Dataset.load_from_df(self.rating_df[['user_id', 'recipe_id', 'rating']], reader)
        test_df = pd.read_csv(self.config["svd_datasets"]["test"]).dropna(subset=['user_id', 'recipe_id', 'rating'])
        

        print("SVD Recommender model Train Start...")
        self.train(model, train_data)
        print("SVD Recommender model Train End...!\n")

        random_user_id = self.random_user()
        print("Start Recomender recipe for {}...".format(random_user_id))
        
        non_rated_recipes = self.get_non_rated_recipe(random_user_id)
        predictions = [model.predict(str(random_user_id), str(recipe_id)) for recipe_id in non_rated_recipes]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_predictions = predictions[:self.config["target"]["top_n"]]

        top_recipe_ids = [int(pred.iid) for pred in top_predictions]
        top_recipe_rating = [pred.est for pred in top_predictions]
        top_recipe_name = self.recipe_df.set_index('id').loc[top_recipe_ids]['name'].tolist()
        top_recipe_preds = [(id, name, rating) for id, name, rating in zip(top_recipe_ids, top_recipe_name, top_recipe_rating)]

        return model,top_recipe_preds, random_user_id

    def display_recommendations(self):
        model, top_recipe_preds, random_user_id = self.recomm_recipe_by_surprise()
        print('#####Top-10 recommended recipe lists (for user: {}) #####'.format(random_user_id))
        for top_recipe in top_recipe_preds:
            print(top_recipe[1], ":", round(top_recipe[2],4))

        #self.evaluate(model) 
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
        self.k = self.config["user_based"]["k"]
        self.similarity=self.config["user_based"]["similarity"]
    def user_based_cf_model(self):
        trainset = self.data.build_full_trainset()
        sim_options = {
            'name' : self.similarity,
            'user_based': True
        }

        knn_model = KNNBasic(k=self.k, sim_options=sim_options)
        knn_model.fit(trainset)

        print(f"User-Based CF Model trained with k={self.k}, similarity='{self.similarity}'")
        return knn_model, trainset
    def filter_testset(self, test_data, trainset):
        testset_filtered = []
        for (user_id, recipe_id, rating) in test_data:
            try:
                inner_uid = trainset.to_inner_uid(user_id)
                inner_iid = trainset.to_inner_iid(recipe_id)
                testset_filtered.append((inner_uid, inner_iid, rating))
            except:
                continue
        return testset_filtered
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
        
        print("\nTop 10 Recommended Recipes:")
        for i, pred in enumerate(top_n_recommendations, 1):
            recipe_id = int(pred.iid)
            recipe_info = self.recipes_df[self.recipes_df['id'] == recipe_id]
            if not recipe_info.empty:
                name = recipe_info['name'].values[0]
                steps = recipe_info['steps'].values[0]
                print(f"{i}. Recipe ID: {recipe_id}, Name: {name}")
                #print(f"{i}. Recipe ID: {recipe_id}, Name: {name}, Steps: {steps}")
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
        user_ids  = self.ratings_df['user_id'].unique()
        random_user_id = np.random.choice(user_ids)

        user_exists = self.recommend_recipes(user_based_model, random_user_id, trainset)

        if not user_exists:
            print("User ID not found. Showing popular recipes instead.")
            self.recommend_popular_recipe()
    
        self.evaluate(user_based_model, trainset)


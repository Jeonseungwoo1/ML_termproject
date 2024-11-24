# Recipe Recommendation System
This project is a recipe recommendation system

## Requirements

### 1. Git Clone
#### 1.1. Clone this Repository
```angular2html
git clone https://github.com/Jeonseungwoo1/ML_termproject.git
```
#### 1.2. File Format
```angular2html
├── config
│   └──config.json
│
├── utils
│   ├── experiment.py
│   ├── file_unzip.py
│   ├── data_loader.py
│   └── hyperparameter_tuning.py
│
├── recommender
│   └── recommender.py
│
├── main.py
└── requirements.txt  
```
### 2. Install Requirements.txt
Python == 3.12.4 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
#### 1.1. Food.com Recipe & Review Data
The original data can be downloaded in following link:
* Food.com Recipe & Review Dataset - [Link (Original)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
The Sub Dataset's original data  can be downloaded in following link:
* MovieLens Latest Datasets - [Link (Original)](https://grouplens.org/datasets/movielens/)


You can download the required dataset files from the following Google Drive link:
* Food.com Recipe & Review Datasets(Include Preprocessed datasets) - [Link (Download Datasets from Google Drive)](https://drive.google.com/drive/folders/1TRZ-GuDaqjtYO-CWX0YhoXU5zrpC4dY9?usp=drive_link)
* Please ensure these files are placed in the `./datasets` directory within your project structure.


#### 1.2. Unzipping Data Files
After download datasets.zip from goole drive link, run this code for unzip .zip file
```angular2html
python utils/file_unzip.py
```

#### 1.3. Format of Datasets
```angular2html
├── datasets
    ├── content_based_train_data.csv
    ├── content_based_test_data.csv
    ├── interactions_train.csv
    ├── interactions_test.csv
    ├── interactions_validation.csv
    ├── PP_interactions_train.csv
    ├── RAW_recipes.csv
    ├── RAW_interactions.csv
    └── Remove_Header_interaction_train.csv
```

#### 1.4. Illustration of Datasets
- `RAW_recipes.csv`: Contains information about recipes.
- `interactions_train.csv`: Contains user interactions (ratings) with recipes before preprocessing.
- `PP_interactions_train.csv`: Contains user interactions (ratings) with recipes.
  * This file has been formatted with columns named `user`, `item`, and `rating` to align with the format required by the `surprise` library's `Reader`.
 - `Remove_Header_interaction_train.csv`: A version of the `PP_interactions_train.csv` with the header removed for specific processing needs.
   

### 2. Program Operation
#### 2.1. Set Target User's Information
- You can change target user's informations by changing `config.json` file 
- If you want run SVD recommender, please change `config.json` file's "setting":{"algo": "..."} to SVD
- If you want run UserBased recommender, please change `config.json` file's "setting":{"algo": "..."} to "User_based"
- If you want run ItemBased recommender, please change `config.json` file's "setting":{"algo": "..."} to "Item_based"
- If you want run ContentBased recommender, please change `config.json` file's "setting":{"algo": "..."} to "Content_based"
- If you wnad run TruncatedSVD recommender, pleas change `config.json` file's "setting":{"algo": "..."} to "TruncatedSVD"
- 
#### 2.2. Set Dataset
- You can choose datasets between "Recipe & Review dataset" and "MovieLens dataset"
- If you set the `config.json`file's "movie":{"use" : "True"}, you can use Movielens Dataset
- If you set the `config.json`file's "movie":{"use" : "False"}, you can use Recipe & Reivew Dataset

#### 2.3. Start Recommender System
```angular2html
python main.py -c ./config/config.json
```

## Result
### SVD
#### Result
<img width="600" src=https://github.com/user-attachments/assets/15c8835b-92e8-41a0-9f8d-68406d9209d4>

#### Performance
<img width="600" src=https://github.com/user-attachments/assets/69080434-6683-45dd-a135-039bd3f39862>

### User Based
#### Result
<img width="600" src=https://github.com/user-attachments/assets/e85951f7-643c-4513-af1d-b60e22494ca6>

#### Performance
<img width="600" src=https://github.com/user-attachments/assets/b36acbb7-fe3c-4ead-8bf7-f46c0e06a07d>

### Item Based
#### Result
<img width="600" src=https://github.com/user-attachments/assets/7c41fa98-8a34-4ff0-b0fc-579be7f5c6be>

#### Performance
<img width="600" src="https://github.com/user-attachments/assets/added657-891f-4829-ac73-d2fcc7eac306">

### Content Based
#### Result
<img width="600" src=https://github.com/user-attachments/assets/62dbacc9-724d-43a8-a0d2-655f166bdf79>

## Reference

* [Surprise](https://surpriselib.com/)
* [Reference Research Paper Link: “Generating Personalized Recipes from Historical User Preferences”](https://arxiv.org/abs/1909.00105)
* [DZone](https://dzone.com/articles/a-deep-dive-into-recommendation-algorithms-with-ne)
* [Machine Learning - Netflix movie recommendation system](https://anil-iqbal.github.io/netflix/)
#### Dataset Reference:

* [Food.com Recipe & Review Data](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
* [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

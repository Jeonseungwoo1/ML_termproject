# Recipe Recommendation System
This project is a recipe recommendation system that uses collaborative filtering to suggest recipes to users based on their past interactions.

<img width="600" alt="스크린샷 2024-11-06 오전 9 22 15" src="https://github.com/user-attachments/assets/d8d524dc-4d88-4d8b-af4f-21d9ef64080e">


## Requirements

Python == 3.12.4 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```


## Usage

### 1. Data Preparation
#### 1.1. Food.com Recipe & Review Data
The original data can be downloaded in following link:
* Food.com Recipe & Review Dataset - [Link (Original)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)


You can download the required dataset files from the following Google Drive link:
* Food.com Recipe & Review Datasets(Include Preprocessed datasets) - [Link (Download Datasets from Google Drive)](https://drive.google.com/drive/folders/1TRZ-GuDaqjtYO-CWX0YhoXU5zrpC4dY9?usp=drive_link)
* Please ensure these files are placed in the `./datasets` directory within your project structure.


#### 1.2. Unzipping Data Files
```angular2html
python utils/file_unzip.py
```

#### 1.3. Format of Datasets
```angular2html
├── datasets
    ├── interactions_train.csv
    ├── PP_interactions_train.csv
    ├── RAW_recipes.csv
    └── Remove_Header_interaction_train.csv
```

#### 1.4. Illustration of Datasets
- `RAW_recipes.csv`: Contains information about recipes.
- `interactions_train.csv`: Contains user interactions (ratings) with recipes before preprocessing.
- `PP_interactions_train.csv`: Contains user interactions (ratings) with recipes.
  * This file has been formatted with columns named `user`, `item`, and `rating` to align with the format required by the `surprise` library's `Reader`.
 - `Remove_Header_interaction_train.csv`: A version of the `PP_interactions_train.csv` with the header removed for specific processing needs.
   
### 2. Pre-Work
#### 2.1. Hyperparameters Tuning
```angular2html
python3 ./utils/hyperparameter_tuning.py -c ./config/config.json
```

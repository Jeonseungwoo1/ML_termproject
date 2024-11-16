# Recipe Recommendation System
This project is a recipe recommendation system

#### Result of SVD recommender
<img width="600" src=https://github.com/user-attachments/assets/9d1f71de-a13d-42a2-9815-64cc55467132>

#### Performance of SVD recommender
<img width="600" src=https://github.com/Jeonseungwoo1/ML_termproject/issues/5#issue-2663747047>

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
│   └── hyperparameter_tuning.py
│
├── main.py
│
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
- You can get best parameters by using this code
- If you want test another candidates, change "param_grid" in `config.json`
```angular2html
python ./utils/hyperparameter_tuning.py -c ./config/config.json
```

- If you got the best parameters, edit “parameter” in ```config.json```
### 3. Program Operation
#### 3.1. Set Target User's Information
- You can change target user's informations by changing `config.json` file 

#### 3.2. Start Recommender System
```angular2html
python main.py -c ./config/config.json
```

## Reference

* [Surprise](https://surpriselib.com/)

# EBA5008-Home-Credit

## Introduction

In an effort to refine the assessment of creditworthiness, this project targets the Home Credit default risk challenge, with a focus on developing a robust machine learning model capable of predicting the probability of default among loan applicants. Our comprehensive approach involved stacking various predictive models, informed post-processing techniques based on historical default data, and a keen utilization of alternative data to address the lack of traditional credit history. Achieving an AUC score of 0.802, the model significantly outperformed the baseline provided by Kaggle. Recommendations for future work include exploring additional data sources and maintaining a strong commitment to the transparency and explainability of the model, thereby contributing to the reduction of financial exclusion and the promotion of responsible credit distribution.

## Installation

To use this project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:Viva-La-Vida2020/EBA5008-Home-Credit.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd EBA5008-Home-Credit
   ```

3. **Install Dependencies:** This project requires several Python libraries. You can install them using pip, the Python package manager. Run the following command to install the required dependencies:

   ```bash
   pip install xgboost lightgbm scikit-learn tensorflow
   ```

## Data

### Raw Data

You can find the raw data at https://www.kaggle.com/c/home-credit-default-risk/data. You can download the data diretly in Kaggle page, but we recommend you to download using kaggle CLI. Follow these steps to download the dataset:

1. **Install the Kaggle CLI:**
   If you haven't already installed the Kaggle CLI, you can do so using pip, the Python package manager:
   
   ```bash
   pip install kaggle
   ```
   
2. **Configure the Kaggle API:**
   Before you can download datasets from Kaggle, you need to configure the Kaggle API by obtaining an API key. Follow these steps:
   - Go to the Kaggle website and sign in to your account.
   - Navigate to the "Account" tab of your user profile.
   - Scroll down to the "API" section and click on "Create New API Token".
   - This will download a file named `kaggle.json`. Save this file in the `~/.kaggle/` directory on your local machine.
   - You can also manually create the `~/.kaggle/` directory if it doesn't exist.

3. **Download the Dataset:**
   Once the Kaggle CLI is installed and the API is configured, you can use the `kaggle datasets download` command to download the dataset. 
   
   ```bash
   kaggle competitions download -c home-credit-default-risk
   ```
   
4. **Extract the Dataset:**
   After downloading the dataset, it will be in a compressed format (e.g., ZIP or TAR). You can use a file extraction tool or command-line utility to extract the contents of the downloaded file.

### Processed Data

You can run the code in src/code/FeatureEngineering to process the data, we also  provide the processed data for you. Download it from https://drive.google.com/file/d/1ldo_b-TuOs9nxQpQ86z5ktzA1GGRySeh/view?usp=drive_link. 

## Usage

Inside the `src/code` directory, you'll find the following subdirectories:

```
src/code
├── EDA
│   ├── (EDA code files)
├── FeatureEngineering
│   ├── (Feature engineering code files)
├── Modeling
│   ├── (Modeling code files)
```

- `EDA`: Contains code for conducting exploratory data analysis.
- `FeatureEngineering`: Contains code for feature engineering.
- `Modeling`: Contains code for building machine learning models.

Explore the code in each subdirectory to understand how different parts of the project are implemented. You can run individual scripts or notebooks in each directory as needed.

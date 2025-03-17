import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the training dataset
train_df = pd.read_csv("/Users/alexleslie/Desktop/UNSW_NB15_training-set.csv")

# Filter for DoS attacks and normal traffic in the training dataset
train_df = train_df[(train_df['attack_cat'] == 'DoS') | (train_df['attack_cat'] == 'Normal')]

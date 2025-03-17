import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the training dataset
train_df = pd.read_csv("/Users/alexleslie/Desktop/UNSW_NB15_training-set.csv")

# Filter for DoS attacks and normal traffic in the training dataset
train_df = train_df[(train_df['attack_cat'] == 'DoS') | (train_df['attack_cat'] == 'Normal')]

# Feature engineering for training
train_df['packet_rate'] = train_df['sttl'] / (train_df['dur'] + 1)
train_df['byte_rate'] = train_df['sbytes'] / (train_df['dur'] + 1)

# Define features and target for training
X_train = train_df[['packet_rate', 'byte_rate', 'proto', 'service']]  # Add more features as needed
y_train = train_df['attack_cat']

# Encode categorical variables for training
X_train = pd.get_dummies(X_train, columns=['proto', 'service'])

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Load the testing dataset
test_df = pd.read_csv("/Users/alexleslie/Desktop/UNSW_NB15_testing-set.csv")


# Filter for DoS attacks and normal traffic in the testing dataset
test_df = test_df[(test_df['attack_cat'] == 'DoS') | (test_df['attack_cat'] == 'Normal')]

# Feature engineering for testing
test_df['packet_rate'] = test_df['sttl'] / (test_df['dur'] + 1)
test_df['byte_rate'] = test_df['sbytes'] / (test_df['dur'] + 1)

# Define features and target for testing
X_test = test_df[['packet_rate', 'byte_rate', 'proto', 'service']]  # Add more features as needed
y_test = test_df['attack_cat']

# Encode categorical variables for testing
X_test = pd.get_dummies(X_test, columns=['proto', 'service'])

# Ensure the feature columns in training and testing match
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Evaluate the model on the testing dataset
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
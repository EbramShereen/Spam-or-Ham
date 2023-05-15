import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Read the data from the CSV file
data = pd.read_csv('spam.csv')

# Extract the features (messages) and labels (categories)
X = data['Message']
y = data['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Convert text messages into a matrix of token counts
vectorizer = CountVectorizer()  
x_train_count = vectorizer.fit_transform(X_train)  
x_test_count = vectorizer.transform(X_test)

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Train the classifier on the training data
dt.fit(x_train_count, y_train)

# Predict the labels for the training data
y_pred = dt.predict(x_train_count)

# Count the number of spam and ham messages
spam_count = pd.Series(y_pred).value_counts()['spam']
ham_count = pd.Series(y_pred).value_counts()['ham']
print("Number of spam:", spam_count)
print("Number of ham:", ham_count)

# Calculate the accuracy of the model on the test data
accuracy = dt.score(x_test_count, y_test)
new_accuracy = accuracy * 100
print("Accuracy:", new_accuracy)

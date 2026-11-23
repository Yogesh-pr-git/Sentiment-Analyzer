# 1. Import Libraries
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# 2. Load and Prepare Dataset
df = pd.read_csv("dataset.csv")
df.drop("review_id", axis=1, inplace=True, errors='ignore')

X_raw = df['review_text']
y = df['sentiment']

# 3. Text Preprocessing
# Step 1: Lowercasing
X_clean = X_raw.str.lower()

# Step 2: Remove Punctuation
X_clean = X_clean.str.replace(r'[^\w\s]', '', regex=True)

# Step 3: Tokenize
tokens = X_clean.str.split()

# Step 4: Remove Stopwords
stop_words = set(stopwords.words('english'))
tokens = tokens.apply(lambda x: [word for word in x if word not in stop_words])

# Step 5: Stemming
stemmer = PorterStemmer()
tokens = tokens.apply(lambda x: [stemmer.stem(word) for word in x])

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert cleaned tokens back to strings (sentences) for Pipeline input
X_processed_text = tokens.apply(lambda x: ' '.join(x))

# Split raw text dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_text, y, test_size=0.2, random_state=42)

# 4. Define and Evaluate Pipelines (BoW vs TF-IDF Comparison with Naive Bayes)
# A. Bag of Words (CountVectorizer) Pipeline
bow_nb_pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('classifier', MultinomialNB())
])

bow_nb_pipeline.fit(X_train, y_train)
bow_pred = bow_nb_pipeline.predict(X_test)
bow_prob = bow_nb_pipeline.predict_proba(X_test)

print("=== Multinomial Naive Bayes (Bag of Words) Pipeline Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, bow_pred):.4f}")
print(f"Precision: {precision_score(y_test, bow_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, bow_pred, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, bow_pred, average='weighted'):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, bow_prob, average='weighted', multi_class='ovr'):.4f}\n")

# B. TF-IDF (TfidfVectorizer) Pipeline
tfidf_nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

tfidf_nb_pipeline.fit(X_train, y_train)
tfidf_pred = tfidf_nb_pipeline.predict(X_test)
tfidf_prob = tfidf_nb_pipeline.predict_proba(X_test)

print("=== Multinomial Naive Bayes (TF-IDF) Pipeline Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, tfidf_pred):.4f}")
print(f"Precision: {precision_score(y_test, tfidf_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, tfidf_pred, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, tfidf_pred, average='weighted'):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, tfidf_prob, average='weighted', multi_class='ovr'):.4f}\n")

# 5. Define Random Forest Pipeline for Cross-Validation and Tuning (using TF-IDF)
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("=== Random Forest (TF-IDF) Pipeline Cross-Validation ===")
# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(rf_pipeline, X_processed_text, y, cv=kfold, scoring='accuracy')
print(f"K-Fold Scores: {scores_kfold}")
print(f"K-Fold Mean Accuracy: {scores_kfold.mean():.4f}")

# Stratified K-Fold Cross Validation
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skfold = cross_val_score(rf_pipeline, X_processed_text, y, cv=skfold, scoring='accuracy')
print(f"Stratified K-Fold Scores: {scores_skfold}")
print(f"Stratified K-Fold Mean Accuracy: {scores_skfold.mean():.4f}\n")

# 6. Hyperparameter Tuning using GridSearchCV on Pipeline
print("=== Hyperparameter Tuning with GridSearchCV on Pipeline ===")
# Define the parameter grid (using pipeline estimator syntax '__')
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_processed_text, y)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

print("=====Model Serialization=====")

#Save the Model
import joblib
joblib.dump(tfidf_nb_pipeline,'model.pkl')
print("Model Successfully Saved!")

#Load the Saved Model
loaded_pipeline = joblib.load('model.pkl')
prediction = loaded_pipeline.predict(X_test)
print("Model Successfully Loaded")
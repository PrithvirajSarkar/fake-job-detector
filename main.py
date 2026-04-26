import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import joblib


# STEP 1: Load dataset
print("Loading dataset...")
df = pd.read_csv("fake_job_postings.csv", encoding='latin1')

print("\nFirst rows:")
print(df.head())


# STEP 2: Check class distribution
print(df["fraudulent"].value_counts())


# STEP 3: Handle missing values
print("\nHandling missing values...")

df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["requirements"] = df["requirements"].fillna("")


# STEP 4: Combine text
print("\nCombining text...")

df["text"] = df["title"] + " " + df["description"] + " " + df["requirements"]


# STEP 5: Convert text to numbers
print("\nConverting text to numbers...")

vectorizer = CountVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["text"])
y = df["fraudulent"]


# STEP 6: Balance dataset (IMPORTANT)
print("\nBalancing dataset...")

df["text_vector"] = list(X.toarray())

df_fake = df[df["fraudulent"] == 1]
df_real = df[df["fraudulent"] == 0].sample(len(df_fake), random_state=42)

df_balanced = pd.concat([df_fake, df_real])
df_balanced = df_balanced.sample(frac=1, random_state=42)

X_balanced = list(df_balanced["text_vector"])
y_balanced = df_balanced["fraudulent"]


# STEP 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)


# STEP 8: Logistic Regression
print("\nTraining Logistic Regression...")

model1 = LogisticRegression(max_iter=1000, class_weight='balanced')
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

# STEP 9: Naive Bayes
print("\nTraining Naive Bayes...")

model2 = MultinomialNB()
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

# STEP 10: Choose best model
best_model = model1


# STEP 11: Save model
joblib.dump(best_model, "job_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved!")


# STEP 12: User input
user_text = input("Enter job description: ")

user_vec = vectorizer.transform([user_text])

prediction = best_model.predict(user_vec)

if prediction[0] == 1:
    print("Fake Job Posting")
    print("Suggestion: Avoid jobs with unrealistic salary or no experience requirement.")
else:
    print("Real Job Posting")
    print("Suggestion: This job looks legitimate based on given information.")
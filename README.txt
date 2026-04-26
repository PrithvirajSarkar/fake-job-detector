FAKE JOB POSTING DETECTION SYSTEM

Project Description:
This project is a Machine Learning-based system used to detect whether a job posting is fake or real. It analyzes job descriptions and predicts the result using trained models.

Dataset:
The dataset used is "fake_job_postings.csv", which contains job-related information such as title, description, requirements, and a target column "fraudulent".

Steps Performed:
1. Loaded the dataset using pandas
2. Checked for missing values and handled them
3. Combined text features (title, description, requirements)
4. Converted text into numerical form using CountVectorizer
5. Split data into training and testing sets
6. Trained two models:
   - Logistic Regression
   - Naive Bayes
7. Compared model accuracy
8. Selected the best model
9. Saved the model using joblib
10. Created a user input system for prediction

Model Used:
Logistic Regression (final selected model)

How to Run:
1. Make sure all files are in the same folder:
   - main.py
   - fake_job_postings.csv
   - job_model.pkl
   - vectorizer.pkl

2. Run the program:
   python main.py

3. Enter a job description when prompted

4. Output will show:
   - Fake Job Posting
   - Real Job Posting


Limitations:
The model is based on text patterns and may not correctly classify all realistic fake job postings.

Conclusion:
The model predicts job postings based on text patterns. It works well for obvious fake jobs but may not be perfect for all cases.

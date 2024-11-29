
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

email_data = {
    "message": [
        "Win $1000 now!", 
        "Your bank account needs verification", 
        "Meeting at 10 am", 
        "Congrats, you won a prize", 
        "Let's catch up tomorrow"
    ],
    "category": ["Spam", "Spam", "Ham", "Spam", "Ham"]
}

df = pd.DataFrame(email_data)

vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(df['message'])
y_labels = df['category']

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_features, y_labels, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train_set, y_train_set)

predictions = classifier.predict(X_test_set)

print(classification_report(y_test_set, predictions))

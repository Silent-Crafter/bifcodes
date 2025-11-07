import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('./sms.tsv', delimiter='\t', header=None, names=['label', 'message'])

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df['message']
y = df['label']

print(df.isnull().sum())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

model = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    MultinomialNB()
)
model.fit(X_train, y_train)
model.fit

pred = model.predict(X_test)
print(classification_report(y_test, pred))

samples = [
    "Win a free iPhone! Click here to claim your prize now.",
    "Are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Reset your password immediately!",
    "Hey, just checking in — how's your day going?"
]

predictions = model.predict(samples)
for msg, label in zip(samples, predictions):
    print(f"\nMessage: {msg}\nPredicted: {'SPAM' if label==1 else 'HAM'}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already present
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('test.csv', encoding='ISO-8859-1')

# Drop rows with missing values
df.dropna(inplace=True)

# Clean column names (optional)
df.columns = df.columns.str.strip()

# Clean text column
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['text'].astype(str).str.lower().apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

plt.figure(figsize=(18, 6))

# Plot 1: Sentiment Distribution
plt.subplot(1, 3, 1)
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Plot 2: Top 10 Countries
top_countries = df['Country'].value_counts().head(10)

plt.figure(figsize=(10,6))
plt.pie(
    top_countries.values,
    labels=top_countries.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('Set3')
)
plt.title('Top 10 Countries by Tweet Count')
plt.ylabel('')  # Remove y-label
plt.show()


# Plot 3: Average Sentiment by Age Group (encoded)
# Convert sentiment to numeric for analysis: positive=1, neutral=0, negative=-1
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)

plt.subplot(1, 3, 3)
age_order = df['Age of User'].value_counts().index  # Sorted by frequency
sns.barplot(x='sentiment_score', y='Age of User', data=df, estimator='mean',
            order=age_order, palette='coolwarm')
plt.title('Avg Sentiment Score by Age Group')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Age Group')

plt.tight_layout()
plt.show()
X = df['clean_text']
y = df['sentiment']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

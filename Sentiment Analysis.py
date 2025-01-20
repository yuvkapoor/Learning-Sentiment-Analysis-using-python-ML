import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Read and preprocess the text
text = open('/content/drive/MyDrive/Colab Notebooks/test.txt', encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Tokenize words
tokenized_words = word_tokenize(cleaned_text, "english")

# Remove stopwords
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Lemmatization
lemma_words = []
lemmatizer = WordNetLemmatizer()
for word in final_words:
    word = lemmatizer.lemmatize(word)
    lemma_words.append(word)

# Emotion detection
emotion_list = []
with open('/content/drive/MyDrive/Colab Notebooks/emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        if word in lemma_words:
            emotion_list.append(emotion)

# Count emotions
print("Detected Emotions:", emotion_list)
emotion_counts = Counter(emotion_list)
print("Emotion Counts:", emotion_counts)

# Sentiment analysis
def sentiment_analyse(sentiment_text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(sentiment_text)
    print("Sentiment Scores:", score)

    if score['neg'] > score['pos']:
        print("Overall Sentiment: Negative")
    elif score['neg'] < score['pos']:
        print("Overall Sentiment: Positive")
    else:
        print("Overall Sentiment: Neutral")

    # Create a bar chart for sentiment scores
    plt.figure(figsize=(8, 6))
    plt.bar(score.keys(), score.values(), color=['blue', 'green', 'red', 'purple'])
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Categories')
    plt.ylabel('Scores')
    plt.show()

# Perform sentiment analysis and visualize results
sentiment_analyse(cleaned_text)

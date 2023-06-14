import string
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

text = open('/content/drive/MyDrive/Colab Notebooks/test.txt', encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

tokenized_words = word_tokenize(cleaned_text, "english")

final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

emotion_list = []
with open('/content/drive/MyDrive/Colab Notebooks/emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)

print(emotion_list,'\n')
w = Counter(emotion_list,),'\n'
print(w,'\n')


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment",'\n')
    elif score['neg'] < score['pos']:
        print("Positive Sentiment",'\n')
    else:
        print("Neutral Sentiment",'\n')


sentiment_analyse(cleaned_text)

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# creating object of SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# loading csv file, using specific columns.
df = pd.read_csv('/Users/abiodunobafemi/Downloads/new_master_USA.csv', usecols=['Text','States_Abrv', 'Created_at'])
df.head()

# Show all vader results -> negative, neutral, positive, compound (computed by normalizing the scores neg, neu, pos)
df['VaderScores'] = df['Text'].apply(lambda Text:sid.polarity_scores(Text))
# Show only compound
df['Compound'] = df['Text'].apply(lambda Text:sid.polarity_scores(Text)['compound'])
df.head()

#print to csv
df.to_csv("Covid_Vader.csv", index=False)
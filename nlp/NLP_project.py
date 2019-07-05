# NLP 
# Princeton School of AI - 2019-03-06/Meetup at Princeton Library of New Jersey
# Vinicius Pantoja/
#

import spacy          #use NLP functions
import tweepy         #pip install -U tweepy
import re
import en_core_web_sm #(python -m spacy download en_core_web_lg)
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
from spacy import displacy

nlp = en_core_web_sm.load()

doc1 = nlp(u'NJ Princeton Public Library is awesome. Enter https://princetonlibrary.org/')

for token in doc1:
    print(token.text)
    
displacy.render(doc1, style='ent',jupyter = True)

displacy.render(doc1, style='dep',jupyter = True)

token_exemple = doc1[0]
print(token_exemple)
print(token_exemple.ent_type_)
print(token_exemple.is_stop)

[x for x in doc1 if x.text not in spacy.lang.en.stop_words.STOP_WORDS]
#
spacy.lang.en.stop_words.STOP_WORDS

w = 'test'
nlp.vocab[w].is_stop = True
print(nlp('test')[0].is_stop)

print(token_exemple.vector[:10])
len(token_exemple.vector)

doc2 = nlp('I am feeling blue')

doc3 = nlp('the sky is blue')

print(doc2[-1].vector[0])
print(doc3[-1].vector[0])

#https://projector.tensorflow.org/

doc4 = nlp('I am sad')

print('The similarity between doc 2 and doc 3 is {}'.format(doc2.similarity(doc3)))
print('The similarity between doc 2 and doc 4 is {}'.format(doc2.similarity(doc4)))

print("Spacy's sentiment analysis result is {}".format(doc4.sentiment))

print("TextBlob's sentiment analysis result is {}".format(TextBlob('I am sad').sentiment))

#token_exemple
#using twitter
#functions
#Princeton School of AI
#
class Connection_twitter():
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, keyword):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.key_word = [keyword]

        return None
        
#%%
# Structure the tweets

    def get_info(self,tweet):
        id = tweet.id
        text_ = tweet.full_text
        author = tweet.author
        date = tweet.created_at
        location = tweet.coordinates
        favorite_count = tweet.favorite_count

        user_favorite_count = tweet.user.favourites_count
        user_followers_count = tweet.user.followers_count
        user_location = tweet.user.location
		
        return id, text_, author, date, location, favorite_count, user_favorite_count, user_followers_count, user_location


    def get_texts(self):

        all_results = []
        for i in range(len(self.key_word)):
            tweets = tweepy.Cursor(self.api.search, q=self.key_word, lang = "en",tweet_mode='extended').items(100)
            result = [tweet for tweet in tweets]
            all_results.append(result)

        for i in range(len(all_results)):
            if i == 0:
                tweet_data = pd.DataFrame([self.get_info(tweet) for tweet in all_results[i]])

            else:
                tweet_data = pd.concat([pd.DataFrame([self.get_info(tweet) for tweet in all_results[i]]), tweet_data])

        tweet_data.columns = ['id', 'text_', 'author', 'date', 'location', 'favorite_count', 'user_favorite_count', 'user_followers_count', 'user_location']

        return tweet_data
        
        
# tweeter analysis
# get the following keys from twitter developer site
#
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
<<<<<<< HEAD
<<<<<<< HEAD
#keyword = ''
=======
=======
>>>>>>> 8ecd5d8... update
#keyword = 'harmony'
>>>>>>> c0941a328fb8882a65efc8bd8e38445879662e9a

#
# list of keywords
#

keyword = ["AI", "Machine Learning", "Princeton"]
#keyword = ["MD", "Cardiovascular", "Medicare"]
twitter = Connection_twitter(consumer_key,consumer_secret,access_token,access_token_secret,keyword)

data = twitter.get_texts()

#https://developer.twitter.com/content/developer-twitter/en.html

#pd column format
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 1000)
#pd.set_option('display.width', 1000)

print(data.shape[0])
print(data.shape[1])
print("Data types\n", data.dtypes)
print(data.head(20))


#save df into a json file
data.to_json('./data/NLP_project.json')

print(data['text_'][1])
print(data['author'][1])

data['Sentiment'] = [TextBlob(x).sentiment[0] for x in data['text_'] ]
data['Subjectivity'] = [TextBlob(x).sentiment[1] for x in data['text_'] ]
print(data.head(10))

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(data['text_'][0])

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#...
text_pro = ''
for text in data[data['Sentiment']>0]['text_']:
  
  text_pro = text_pro + ' ' + text

text_pro

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_pro)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#...
text_negative = ''
for text in data[data['Sentiment']<0]['text_']:
  
  text_negative = text_negative + ' ' + text

text_negative

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_negative)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#...
def clean_text(text_series):
  nlp = en_core_web_sm.load()
  full_text = ''
  
#  new_stop_words = ['RT','Machine Learning']
  new_stop_words = ['RT','AI']
  for word in new_stop_words:
    nlp.vocab[word].is_stop = True
  
  for text in text_series:
    text1 = nlp(text)
    
    for token in text1:
      if token.is_stop:
        pass
      else:
        full_text = full_text + ' ' + token.text

  return full_text

text_positive = clean_text(data[data['Sentiment']>0]['text_'])

print(text_negative)

##..
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_positive)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# end of code

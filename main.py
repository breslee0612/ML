import pandas as pd
import nltk
import string
import ast
import re
import unidecode
import streamlit as st
from PIL import Image
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


class config():
  RECIPES_PATH = 'df_recipes.csv'
  PARSED_PATH = 'df_parsed.csv'
  MODEL_PATH = 'model_cbow.bin'


def ingredient_parser(ingreds):
  measures = [
    'teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbl.', 'tb', 'tbsp.',
    'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 'fl pt',
    'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml', 'milliliter',
    'millilitre', 'cc', 'mL', 'l', 'liter', 'litre', 'L', 'dl', 'deciliter',
    'decilitre', 'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch',
    'medium', 'slice', 'pound', 'lb', '#', 'ounce', 'oz', 'mg', 'milligram',
    'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'x',
    'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre',
    'm', 'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto',
    'kilo'
  ]
  words_to_remove = [
    'fresh', 'oil', 'a', 'red', 'bunch', 'and', 'clove', 'or', 'leaf',
    'chilli', 'large', 'extra', 'sprig', 'ground', 'handful', 'free', 'small',
    'pepper', 'virgin', 'range', 'from', 'dried', 'sustainable', 'black',
    'peeled', 'higher', 'welfare', 'seed', 'for', 'finely', 'freshly', 'sea',
    'quality', 'white', 'ripe', 'few', 'piece', 'source', 'to', 'organic',
    'flat', 'smoked', 'ginger', 'sliced', 'green', 'picked', 'the', 'stick',
    'plain', 'plus', 'mixed', 'mint', 'bay', 'basil', 'your', 'cumin',
    'optional', 'fennel', 'serve', 'mustard', 'unsalted', 'baby', 'paprika',
    'fat', 'ask', 'natural', 'skin', 'roughly', 'into', 'such', 'cut', 'good',
    'brown', 'grated', 'trimmed', 'oregano', 'powder', 'yellow', 'dusting',
    'knob', 'frozen', 'on', 'deseeded', 'low', 'runny', 'balsamic', 'cooked',
    'streaky', 'nutmeg', 'sage', 'rasher', 'zest', 'pin', 'groundnut',
    'breadcrumb', 'turmeric', 'halved', 'grating', 'stalk', 'light', 'tinned',
    'dry', 'soft', 'rocket', 'bone', 'colour', 'washed', 'skinless',
    'leftover', 'splash', 'removed', 'dijon', 'thick', 'big', 'hot', 'drained',
    'sized', 'chestnut', 'watercress', 'fishmonger', 'english', 'dill',
    'caper', 'raw', 'worcestershire', 'flake', 'cider', 'cayenne', 'tbsp',
    'leg', 'pine', 'wild', 'if', 'fine', 'herb', 'almond', 'shoulder', 'cube',
    'dressing', 'with', 'chunk', 'spice', 'thumb', 'garam', 'new', 'little',
    'punnet', 'peppercorn', 'shelled', 'saffron', 'other'
    'chopped', 'salt', 'olive', 'taste', 'can', 'sauce', 'water', 'diced',
    'package', 'italian', 'shredded', 'divided', 'parsley', 'vinegar', 'all',
    'purpose', 'crushed', 'juice', 'more', 'coriander', 'bell', 'needed',
    'thinly', 'boneless', 'half', 'thyme', 'cubed', 'cinnamon', 'cilantro',
    'jar', 'seasoning', 'rosemary', 'extract', 'sweet', 'baking', 'beaten',
    'heavy', 'seeded', 'tin', 'vanilla', 'uncooked', 'crumb', 'style', 'thin',
    'nut', 'coarsely', 'spring', 'chili', 'cornstarch', 'strip', 'cardamom',
    'rinsed', 'honey', 'cherry', 'root', 'quartered', 'head', 'softened',
    'container', 'crumbled', 'frying', 'lean', 'cooking', 'roasted', 'warm',
    'whipping', 'thawed', 'corn', 'pitted', 'sun', 'kosher', 'bite', 'toasted',
    'lasagna', 'split', 'melted', 'degree', 'lengthwise', 'romano', 'packed',
    'pod', 'anchovy', 'rom', 'prepared', 'juiced', 'fluid', 'floret', 'room',
    'active', 'seasoned', 'mix', 'deveined', 'lightly', 'anise', 'thai',
    'size', 'unsweetened', 'torn', 'wedge', 'sour', 'basmati', 'marinara',
    'dark', 'temperature', 'garnish', 'bouillon', 'loaf', 'shell', 'reggiano',
    'canola', 'parmigiano', 'round', 'canned', 'ghee', 'crust', 'long',
    'broken', 'ketchup', 'bulk', 'cleaned', 'condensed', 'sherry', 'provolone',
    'cold', 'soda', 'cottage', 'spray', 'tamarind', 'pecorino', 'shortening',
    'part', 'bottle', 'sodium', 'cocoa', 'grain', 'french', 'roast', 'stem',
    'link', 'firm', 'asafoetida', 'mild', 'dash', 'boiling'
  ]
  if isinstance(ingreds, list):
    ingredients = ingreds
  else:
    ingredients = ast.literal_eval(ingreds)
  translator = str.maketrans('', '', string.punctuation)
  lemmatizer = WordNetLemmatizer()
  ingred_list = []
  for i in ingredients:
    i.translate(translator)
    items = re.split(' |-', i)
    items = [word for word in items if word.isalpha()]
    items = [word.lower() for word in items]
    items = [unidecode.unidecode(word) for word in items]
    items = [lemmatizer.lemmatize(word) for word in items]
    items = [word for word in items if word not in measures]
    items = [word for word in items if word not in words_to_remove]
    if items:
      ingred_list.append(' '.join(items))
  ingred_list = " ".join(ingred_list)
  return ingred_list


def get_and_sort_corpus(data):
  corpus_sorted = []
  for doc in data.parsed.values:
    doc = doc.split()
    doc.sort()
    corpus_sorted.append(doc)
  return corpus_sorted


def get_window(corpus):
  lengths = [len(doc) for doc in corpus]
  avg_len = float(sum(lengths)) / len(lengths)
  return round(avg_len)


class TfidfEmbeddingVectorizer(object):

  def __init__(self, word_model):

    self.word_model = word_model
    self.word_idf_weight = None
    self.vector_size = word_model.wv.vector_size

  def fit(self, docs):
    text_docs = []
    for doc in docs:
      text_docs.append(" ".join(doc))
    tfidf = TfidfVectorizer()
    tfidf.fit(text_docs)
    max_idf = max(tfidf.idf_)
    self.word_idf_weight = defaultdict(
      lambda: max_idf,
      [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
    )
    return self

  def transform(self, docs):
    doc_word_vector = self.word_average_list(docs)
    return doc_word_vector

  def word_average(self, sent):
    mean = []
    for word in sent:
      if word in self.word_model.wv.index_to_key:
        mean.append(
          self.word_model.wv.get_vector(word) * self.word_idf_weight[word])
    if not mean:
      return np.zeros(self.vector_size)
    else:
      mean = np.array(mean).mean(axis=0)
      return mean

  def word_average_list(self, docs):
    return np.vstack([self.word_average(sent) for sent in docs])


def title_parser(title):
  title = unidecode.unidecode(title)
  return title


def ingredient_parser_final(ingredient):
  if isinstance(ingredient, list):
    ingredients = ingredient
  else:
    ingredients = ast.literal_eval(ingredient)

  ingredients = ",".join(ingredients)
  ingredients = unidecode.unidecode(ingredients)
  return ingredients


def get_recommendations(N, scores):
  df_recipes = pd.read_csv(config.PARSED_PATH)
  top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
  recommendation = pd.DataFrame(
    columns=['recipe', 'ingredients', 'score', 'url'])
  count = 0
  for i in top:
    recommendation.at[count,
                      'recipe'] = title_parser(df_recipes['recipe_name'][i])
    recommendation.at[count, 'ingredients'] = ingredient_parser_final(
      df_recipes['ingredients'][i])
    recommendation.at[count, 'url'] = df_recipes['recipe_urls'][i]
    recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i]))
    count += 1
  return recommendation


# Assumes `ingreds` is a list of comma-separated ingredient strings
st.title('Recipe Recommendations')
image = Image.open('cooking.jpg')
st.image(image,
         caption='Made by Le Ngoc Gia Huy',
         width=300,
         use_column_width=True)
ingreds = st.text_input('Enter ingredients (separated by commas):')
if st.button('Get recommendation'):
  if not ingreds:
    st.warning('Please enter ingredients')
  else:
    data = pd.read_csv(config.RECIPES_PATH)
    data['parsed'] = data.ingredients.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)
    model = Word2Vec.load(config.MODEL_PATH)
    model.init_sims(replace=True)
    tfidf = TfidfEmbeddingVectorizer(model)
    tfidf.fit(corpus)
    doc_vec = tfidf.transform(corpus)
    doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
    assert len(doc_vec) == len(corpus)
    input = ingreds
    input = input.split(",")
    input = ingredient_parser(input)
    words1 = input.split()
    input_embedding = tfidf.transform([words1])[0].reshape(1, -1)
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0],
                  doc_vec)
    scores = list(cos_sim)
    top_recipes = get_recommendations(5, scores)
    st.write('Here are your top 5 recipe recommendations:')
    st.write(top_recipes)

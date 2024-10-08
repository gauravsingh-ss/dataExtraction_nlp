import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import textstat
nltk.download('punkt')
nltk.download('stopwords')

import os
cwd = os.getcwd()
print("Current working directoryy:", cwd)

input_data = pd.read_excel(cwd+'/Input.xlsx')
output_data = pd.read_excel(cwd+'/Output Data Structure.xlsx')

"""# EXTRACTING CONTENT AND SAVING"""

def extract_article(url):
  r = requests.get(url)
  if r.status_code == 200:
    soup = BeautifulSoup(r.text, 'html.parser')

    article_text = soup.find('div', class_='td-post-content tagdiv-type')
    if ((article_text is None)):
      parent_article_text = soup.find('div', class_='td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type')
      article_text = parent_article_text.find('div', class_ = 'tdb-block-inner td-fix-index')

    paragraphs = article_text.find_all(['p', 'li'])
    text_in_article = ''

    for item in paragraphs:
      if item.name == 'p':
        text_in_article += item.get_text(strip=True) + '\n'
      elif item.name == 'li':
        text_in_article += '-' + item.get_text(strip=True) + '\n'

    try:
      title = soup.find('h1', class_='tdb-title-text').get_text(strip=True)        #extracting article title
    except AttributeError:
      title = soup.find('h1', class_='entry-title').get_text(strip=True)

    return title + '\n' + text_in_article

def saving_txt(text, url_id, url):
  re = requests.get(url)
  if re.status_code == 200:
    if not os.path.exists('txtFile'):
        os.makedirs('txtFile')

    filename = os.path.join('txtFile', f'{url_id}.txt')
    #filename = f'{url_id}.txt'
    with open(filename, 'w') as file:
          file.write(text)

i = 0
while i<input_data.shape[0]:
  url_id = input_data.loc[i, 'URL_ID']
  url = input_data.loc[i, 'URL']

  text = extract_article(url)
  saving_txt(text, url_id, url)

  i+=1

"""# SENTIMENTAL ANALYSIS AND ALSO WORD COUNT"""

def all_stopWords(path):
  all_req_words = set()
  for txt_file in os.listdir(path):
    final_path = os.path.join(path, txt_file)
    with open(final_path, 'r', encoding='latin-1') as f:
      stopwords = set(f.read().splitlines())
      all_req_words.update(stopwords)
  return all_req_words

def cleaning(txt_path, stop_words, i):
  ex_str = set()
  filename = f"blackassign{i:04d}.txt"
  final_path = os.path.join(txt_path, filename)
  if os.path.exists(final_path):
    with open(final_path, 'r') as f:
      contentss = f.read()

    token_words = nltk.word_tokenize(contentss)
    filtered_text = [word for word in token_words if word.upper() not in stop_words]
    ex_str.update(filtered_text)

  return ex_str

# removing the stop words (using stopwords class of nltk package).
# removing any punctuations like ? ! , . from the word before counting.

def cleaning_nltk(txt_path, i):
  ex_str = set()
  filename = f"blackassign{i:04d}.txt"
  final_path = os.path.join(txt_path, filename)
  if(os.path.exists(final_path)):
    with open(final_path, 'r') as f:
      contents = f.read()

    # removing punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    token_words = tokenizer.tokenize(contents)

    sw_nltk = stopwords.words('english')
    filtered_text = [word for word in token_words if word.lower() not in sw_nltk]
    ex_str.update(filtered_text)

  return ex_str

def pos_neg_words(path, pos_neg_txt_filename, cleaned_word):
  given_pos_neg = set()
  final_path = os.path.join(path, pos_neg_txt_filename)
  with open(final_path, 'r', encoding='latin-1') as f:
    pos_neg = set(f.read().splitlines())
    given_pos_neg.update(pos_neg)

  matched_words = [word for word in cleaned_word if word in given_pos_neg]

  return matched_words

def extracting_derived_variables(cleaned_words, positive_word, negative_word):
  pos_score = len(positive_word)
  neg_score = len(negative_word)
  total_words = len(cleaned_words)

  polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)    #Range is from -1 to +1
  subjectivity_score = (pos_score + neg_score) / ((total_words) + 0.000001)          #Range is from 0 to +1

  return pos_score, neg_score, polarity_score, subjectivity_score

"""# ANALYSIS OF READABILITY, COMPLEX WORD COUNT, PRONOUN COUNT, AVERAGE WORD LENGTH"""

def analysis_of_readability(txt_path, i):
  ex_str = set()
  sum_sent = 0
  avg_words_per_sent = 0
  syllable_count_per_word = 0
  per_complex_words = 0
  fog_indx = 0
  complex_word_count = 0
  count_syllable = 0
  pronoun_count = 0
  avg_word_length = 0
  character_count = 0
  avg_sent_length = 0
  filename = f"blackassign{i:04d}.txt"
  final_path = os.path.join(txt_path, filename)
  if os.path.exists(final_path):
    with open(final_path, 'r') as f:
      contents = f.read()

    words = nltk.word_tokenize(contents)
    sentence = nltk.sent_tokenize(contents)
    sum_sent += sum(len(sent) for sent in sentence)

    avg_sent_length = sum_sent / len(sentence)

    sentences_len = textstat.sentence_count(contents)
    words_len = textstat.lexicon_count(contents, removepunct=True)

    # AVG NUMBER OF WORDS PER SENTENCE
    if sentences_len != 0:
      avg_words_per_sent = words_len / sentences_len
    else:
      avg_words_per_sent = 0

    # Complex words are words in the text that contain more than two syllables.
    for word in words:
      syllable_countt = textstat.syllable_count(word)
      count_syllable += syllable_countt
      if(syllable_countt > 2):
        complex_word_count += 1

    #print(count_syllable)
    syllable_count_per_word = count_syllable/words_len
    per_complex_words = complex_word_count / words_len

    #Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
    fog_indx = 0.4 * (avg_words_per_sent + per_complex_words)
    #fog_indx = textstat.gunning_fog(contents)

    # PERSONAL PRONOUN
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(contents)
    pronoun_count = len(pronouns)

    # AVERAGE WORD LENGTH
    # Sum of the total number of characters in each word/Total number of words

    words = cleaning_nltk(article_content_txt_file_path, i)

    character_count = sum(len(word) for word in words)
    avg_word_length = character_count / len(words)

  return avg_words_per_sent, per_complex_words, fog_indx, complex_word_count, pronoun_count, avg_word_length, syllable_count_per_word, avg_sent_length

i=0

StopWords_path = cwd+'/StopWords'
article_content_txt_file_path = cwd+'/txtFile/'

accessed_stop_words = all_stopWords(StopWords_path)

pos_neg_path = cwd+'/MasterDictionary'
pos_filename = 'positive-words.txt'
neg_filename = 'negative-words.txt'

while i<input_data.shape[0]:

  # Cleaning using Stop Words Lists
  cleaned_words = cleaning(article_content_txt_file_path, accessed_stop_words, i+1)

  positive_word = pos_neg_words(pos_neg_path, pos_filename, cleaned_words)
  negative_word = pos_neg_words(pos_neg_path, neg_filename, cleaned_words)

  #output scores of all
  scores = extracting_derived_variables(cleaned_words, positive_word, negative_word)
  #positive
  output_data.loc[i, 'POSITIVE SCORE'] = scores[0]

  #negative
  output_data.loc[i, 'NEGATIVE SCORE'] = scores[1]

  #polarity
  output_data.loc[i, 'POLARITY SCORE'] = scores[2]

  #subjectivity
  output_data.loc[i, 'SUBJECTIVITY SCORE'] = scores[3]

  #Analysis of readability, the below function returns AVG NUMBER OF WORDS PER SENTENCE,
  #PERCENTAGE OF COMPLEX WORDS, FOG INDEX, COMPLEX WORD COUNT, PERSONAL PRONOUNS,
  #AVG WORD LENGTH, SYLLABLE PER WORD, AVG SENTENCE LENGTH

  anlysis = analysis_of_readability(article_content_txt_file_path, i+1)

  output_data.loc[i, 'AVG NUMBER OF WORDS PER SENTENCE'] = anlysis[0]
  output_data.loc[i, 'PERCENTAGE OF COMPLEX WORDS'] = anlysis[1]
  output_data.loc[i, 'FOG INDEX'] = anlysis[2]
  output_data.loc[i, 'COMPLEX WORD COUNT'] = anlysis[3]
  output_data.loc[i, 'PERSONAL PRONOUNS'] = anlysis[4]
  output_data.loc[i, 'AVG WORD LENGTH'] = anlysis[5]
  output_data.loc[i, 'SYLLABLE PER WORD'] = anlysis[6]
  output_data.loc[i, 'AVG SENTENCE LENGTH'] = anlysis[7]

  # word count We count the total cleaned words present in the text by
  # removing the stop words (using stopwords class of nltk package).
  # removing any punctuations like ? ! , . from the word before counting.

  cleaned_words_nltk = cleaning_nltk(article_content_txt_file_path, i+1)
  output_data.loc[i, 'WORD COUNT'] = len(cleaned_words_nltk)

  i+=1

output_data
output_data.to_csv('final_output.csv')
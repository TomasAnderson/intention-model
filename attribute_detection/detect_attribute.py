# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import csv
import pickle
import re
from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
COLOR = ['beige','black','blue','brown','green','grey','navy','orange','pink','purple','red','white','yellow','animal','camouflage','dotted','floral','striped','multi']
MATERIAL = ['cotton','leather','linen','wool','denim','laces','mesh','velvet','satin','faux fur','nylon']
OCCASION = ['casual','work','party','beach','school','maternity','active','wedding','sexy','bohemian']
GENDER = ['men','women','baby','kid']
SEASON = ['spring','summer','autumn','winter']

attr_names = ['ID','genders', 'seasons', 'colors', 'materials', 'occasions', 'brand', 'necks', 'sleeves', 'category']

# genders, seasons, colors, materials, occasions,
#
# brand, ID, necks, sleeves,
#
# price
#
# texts
#
# category


def txt2csv():
    with open("attributes_79061.txt") as f:
        with open("attributes.csv", 'w') as writer:
            results = csv.writer(writer)
            attributes = json.loads(f.readline())
            for item in attributes['products']:
                out = [item['ID'], item['brand'], item['price'],
                item['genders'], item['colors'], item['materials'],
                item['occasions'], item['necks'], item['sleeves'],
                item['texts'], item['category']]
                results.writerow(out)


def inverted_index(attr_name):
    index_dict = {}
    unique_v = 0
    with open("attributes_79061.txt") as f:
        attributes = json.load(f)
        for product in attributes['products']:
            v = product[attr_name]
            if not isinstance(v, (list,)):
                v = [v]
            for elm in v:
                if elm in index_dict:
                    index_dict[elm].append(product['ID'])
                else:
                    unique_v = unique_v + 1
                    index_dict[elm] = [product['ID']]
    print "there are %d unique attribute names in %s" % (unique_v, attr_name)

    with open("./index/"+attr_name+'_index.pkl', 'wb') as f:
        pickle.dump(index_dict, f, pickle.HIGHEST_PROTOCOL)

def attribute_index():
    for attr_name in attr_names:
        inverted_index(attr_name)

def build_tf_idf():
    # test_set = ["The sun in the sky is bright."]  # Query
    print "loading corpus..."
    train_set = load_corpus()
    train_set = train_set[:1000]
    stopWords = stopwords.words('english')


    vectorizer = CountVectorizer(stop_words=stopWords)
    vectorizer.fit(train_set)
    transformer = TfidfTransformer()

    print "vectorising corpus..."
    trainVectorizerArray = vectorizer.transform(train_set).toarray()
    # testVectorizerArray = vectorizer.transform(test_set).toarray()
    # print 'Transform Vectorizer to test set', testVectorizerArray

    print "tfidf transforming..."
    transformer.fit(trainVectorizerArray)

    train_corpus = transformer.transform(trainVectorizerArray).toarray()
    print 'corpus', train_corpus



    # transformer.fit(testVectorizerArray)

    # tfidf = transformer.transform(testVectorizerArray)
    # tfidf = tfidf.todense()

    # test = tfidf
    # print 'result', np.argmax(np.dot(corpus, test.T))
def load_corpus():
    if not path.exists("text_corpus.txt"):
        extract_text()
    with open("text_corpus.txt") as f:
        return f.readlines()

def clean_text(str):
    def rm_repeat_chars(str):
        return re.sub(r'(.)(\1){2,}', r'\1\1', str)

    def rm_time(str):
        return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

    def rm_punctuation(current_tweet):
        return re.sub(r'[^\w\s]', '', current_tweet)

    def rm_digit(str):
        return re.sub("^\d+\s|\s\d+\s|\s\d+$", '', str)

    def rm_char(str):
        return ' '.join([w for w in str.split() if len(w) > 1])

    def rm_underscore(str):
        return ' '.join([w for w in str.split() if '_' not in w])

    str = str.lower()
    str = rm_repeat_chars(str)
    str = rm_time(str)
    str = rm_punctuation(str)
    str = rm_digit(str)
    str = rm_char(str)
    str = rm_underscore(str)
    return str

def extract_text():
    with open("attributes_79061.txt") as f:
        with open("text_corpus.txt", 'w') as writer:
            attributes = json.load(f)
            for product in attributes['products']:
                v = product['texts']
                v = clean_text(v)
                writer.write(v+"\n")



if __name__ == '__main__':
    # Step 1: data preparation

    # genders, seasons, colors, materials, occasions, brand, ID, necks, sleeves, category
    # attribute_index()

    # texts index
    build_tf_idf()

    # price
    exit()


    # Step 2: query
    while True:
        try:
            utterence = raw_input("Please talk to me: ")
            node = detect_attribute(utterence)
            if node :
                print node
            else:
                print("not found")

        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

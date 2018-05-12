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
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from PIL import Image

porter = PorterStemmer()

COLOR = ['beige','black','blue','brown','green','grey','navy','orange','pink','purple','red','white','yellow','animal','camouflage','dotted','floral','striped','multi']
MATERIAL = ['cotton','leather','linen','wool','denim','laces','mesh','velvet','satin','faux fur','nylon']
OCCASION = ['casual','work','party','beach','school','maternity','active','wedding','sexy','bohemian']
GENDER = ['men','women','baby','kid']
SEASON = ['spring','summer','autumn','winter']

attr_names = ['ID','genders', 'seasons', 'colors', 'materials', 'occasions', 'brand', 'necks', 'sleeves', 'category']

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
                elm = porter.stem(elm)
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
    train_set = train_set
    print len(train_set)
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
    print 'corpus', train_corpus.shape



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
    
    tokens = word_tokenize(str)
    str = ' '.join([porter.stem(t) for t in tokens])
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
    def prepare_data():
        attribute_index() # attributes
        # build_tf_idf()    # texts
                          # price
    # prepare_data()

    # step 2: load indexed data

    def load_index(attr):
        with open(path.join('./index', attr+'_index.pkl'), 'rb') as f:
            return pickle.load(f)

    brand_idx = load_index('brand')
    category_idx = load_index('category')
    color_idx = load_index('colors') #
    gender_idx = load_index('genders') #
    material_idx = load_index('materials') #
    necks_idx = load_index('necks')
    occasion_idx = load_index('occasions') #
    season_idx = load_index('seasons') #
    sleeves_idx = load_index('sleeves')



    BRAND, CATEGORY, NECKS, SLEEVES  = [], [], [], []
    for k in brand_idx:
        BRAND.append(k)
    for k in category_idx:
        print k
        CATEGORY.append(k)
    for k in necks_idx:
        NECKS.append(k)
    for k in sleeves_idx:
        SLEEVES.append(k)

    # Step 3: query
    def detect_attr(sent):
        tokens = word_tokenize(sent)
        tokens = [porter.stem(v) for v in tokens]

        # attribute query
        results = []
        intersect_results = set()
        for t in tokens:
            if t in brand_idx:
                print "brand detected"
                results += brand_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(brand_idx[t])
                else:
                    intersect_result = set(brand_idx[t])
            if t in category_idx:
                print "category detected"
                results += category_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(category_idx[t])
                else:
                    intersect_result = set(category_idx[t])
            if t in color_idx:
                print "color detected"
                results += color_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(color_idx[t])
                else:
                    intersect_result = set(color_idx[t])
            if t in gender_idx:
                print "gender detected"
                results += gender_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(gender_idx[t])
                else:
                    intersect_result = set(gender_idx[t])
            if t in material_idx:
                print "material detected"
                results += material_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(material_idx[t])
                else:
                    intersect_result = set(material_idx[t])
            if t in necks_idx:
                print "necks detected"
                results += necks_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(necks_idx[t])
                else:
                    intersect_result = set(necks_idx[t])
            if t in occasion_idx:
                print "occasion detected"
                results += occasion_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(occasion_idx[t])
                else:
                    intersect_result = set(occasion_idx[t])
            if t in season_idx:
                print "season detected"
                results += season_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(season_idx[t])
                else:
                    intersect_result = set(season_idx[t])
            if t in sleeves_idx:
                print "sleeves detected"
                results += sleeves_idx[t]
                if len(intersect_results) > 0:
                    intersect_result.intersection(sleeves_idx[t])
                else:
                    intersect_result = set(sleeves_idx[t])


        # tfidf query


        return list(set(results)), list(intersect_result)
    while True:
        try:
            utterence = raw_input("Please talk to me: ")
            results, intersect_results = detect_attr(utterence)


            if results :
                # print results

                print len(intersect_results)
            else:
                print("not found")

        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

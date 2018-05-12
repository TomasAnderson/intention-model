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
from sklearn.externals import joblib

porter = PorterStemmer()

COLOR = ['beige','black','blue','brown','green','grey','navy','orange','pink','purple','red','white','yellow','animal','camouflage','dotted','floral','striped','multi']
MATERIAL = ['cotton','leather','linen','wool','denim','laces','mesh','velvet','satin','faux fur','nylon']
OCCASION = ['casual','work','party','beach','school','maternity','active','wedding','sexy','bohemian']
GENDER = ['men','women','baby','kid']
SEASON = ['spring','summer','autumn','winter']

attr_names = ['ID','genders', 'seasons', 'colors', 'materials', 'occasions', 'brand', 'necks', 'sleeves', 'category', 'price']

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
        if attr_name == 'ID':
            index_dict = {}
            with open("attributes_79061.txt") as f:
                attributes = json.load(f)
                for i, product in enumerate(attributes['products']):
                    v = product['ID']
                    index_dict[i] = v
            with open("./index/" + attr_name + '_index.pkl', 'wb') as f:
                pickle.dump(index_dict, f, pickle.HIGHEST_PROTOCOL)
        elif attr_name == 'price':
            index_dict = {}
            with open("attributes_79061.txt") as f:
                attributes = json.load(f)
                for product in attributes['products']:
                    v = product['ID']
                    p = product['price']
                    index_dict[v] = p
            with open('./index/id2price_index.pkl', 'wb') as f:
                pickle.dump(index_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            inverted_index(attr_name)

def build_tf_idf():
    # test_set = ["The sun in the sky is bright."]  # Query
    print "loading corpus..."
    train_set = load_corpus()
    train_set = train_set[:1000]
    print len(train_set)
    stopWords = stopwords.words('english')


    vectorizer = CountVectorizer(stop_words=stopWords)
    vectorizer.fit(train_set)
    joblib.dump(vectorizer, "vectorizer.sav")

    transformer = TfidfTransformer()
    print "vectorising corpus..."
    trainVectorizerArray = vectorizer.transform(train_set).toarray()

    print "tfidf transforming..."
    transformer.fit(trainVectorizerArray)
    joblib.dump(transformer, 'tfidf_transformer.sav')

    train_corpus = transformer.transform(trainVectorizerArray).toarray()
    np.save("train_corpus.npy", train_corpus)
    print 'corpus', train_corpus.shape


def load_tfidf_model():
    vectorizer = joblib.load("vectorizer.sav")
    transformer = joblib.load("tfidf_transformer.sav")
    corpus = np.load("train_corpus.npy")
    return vectorizer, transformer, corpus

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

def tfidf_retrieval(query):
    vectorizer, transformer, corpus = load_tfidf_model()

    testVectorizerArray = vectorizer.transform([query]).toarray()
    print 'Transform Vectorizer to test set', testVectorizerArray

    transformer.fit(testVectorizerArray)

    tfidf = transformer.transform(testVectorizerArray)
    tfidf = tfidf.todense()

    test = tfidf
    return id_idx[np.argmax(np.dot(corpus, test.T))]

def extract_price(sent):
    return re.findall('\d+', sent)

def detect_attr(sent):
    tokens = word_tokenize(sent)
    tokens = [porter.stem(v) for v in tokens]

    # attribute query
    results = []
    intersect_result = set()
    for t in tokens:
        if t in brand_idx:
            results += brand_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(brand_idx[t])
            else:
                intersect_result = set(brand_idx[t])
            print "brand detected", len(brand_idx[t]), "results"
        if t in category_idx:
            results += category_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(category_idx[t])
            else:
                intersect_result = set(category_idx[t])
            print "category detected", len(category_idx[t]), "results"
        if t in color_idx:
            results += color_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(color_idx[t])
            else:
                intersect_result = set(color_idx[t])
            print "color detected", len(color_idx[t]), "results"
        if t in gender_idx:
            results += gender_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(gender_idx[t])
            else:
                intersect_result = set(gender_idx[t])
            print "gender detected", len(gender_idx[t]), "results"
        if t in material_idx:
            results += material_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(material_idx[t])
            else:
                intersect_result = set(material_idx[t])
            print "material detected", len(material_idx[t]), "results"
        if t in necks_idx:
            results += necks_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(necks_idx[t])
            else:
                intersect_result = set(necks_idx[t])
            print "necks detected", len(necks_idx[t]), "results"
        if t in occasion_idx:
            results += occasion_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(occasion_idx[t])
            else:
                intersect_result = set(occasion_idx[t])
            print "occasion detected", len(occasion_idx[t]), "results"
        if t in season_idx:
            results += season_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(season_idx[t])
            else:
                intersect_result = set(season_idx[t])
            print "season detected", len(season_idx[t]), "results"
        if t in sleeves_idx:
            results += sleeves_idx[t]
            if len(intersect_result) > 0:
                intersect_result = intersect_result.intersection(sleeves_idx[t])
            else:
                intersect_result = set(sleeves_idx[t])
            print "sleeve detected", len(sleeves_idx[t]), "results"

    return list(set(results)), list(intersect_result)



def search_by_price(price):
    with open("attributes_79061.txt") as f:
        attributes = json.load(f)
        results = []
        for product in attributes['products']:
            v = float(product['price'])
            if abs(price - v) < 50:
                results.append(product['ID'])
        return results


if __name__ == '__main__':
    # Step 1: data preparation

    # genders, seasons, colors, materials, occasions, brand, ID, necks, sleeves, category
    def prepare_data():
        attribute_index() # attributes
        build_tf_idf()    # texts
                          # price


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
    id_idx = load_index('ID')
    id2price_idx = load_index('id2price')

    BRAND, CATEGORY, NECKS, SLEEVES  = [], [], [], []
    for k in brand_idx:
        BRAND.append(k)
    for k in category_idx:
        CATEGORY.append(k)
    for k in necks_idx:
        NECKS.append(k)
    for k in sleeves_idx:
        SLEEVES.append(k)

    # Step 3: query


    while True:
        try:
            utterence = raw_input("Please talk to me: ")
            results, intersect_results = detect_attr(utterence)
            text_results = tfidf_retrieval(utterence)


            if intersect_results != None or text_results != None:
                if intersect_results != None:
                    print len(intersect_results)
                else:
                    print "intersected attribute set size is 0"
                print text_results, "the price is $", id2price_idx[text_results]
            else:
                print("not found")

        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

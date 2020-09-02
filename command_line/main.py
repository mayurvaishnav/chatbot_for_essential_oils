import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
from duckduckpy import query
import sqlite3

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

with open("./../data/intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

try:
    model = tflearn.DNN(net)
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def duckduckgo_response(input_statement):
    try:
        response = query(input_statement, container='dict')
    except:
        return 'Sorry, cannot find anything about it. Please change the query and try again...'

    if response['abstract_text']:
        response_statement = response['abstract_text']
        # response_statement.confidence = 1
    else:
        response_statement = 'Sorry, cannot find anything about it. Please change the query and try again...'
        # response_statement.confidence = 0

    return response_statement

def create_table():
    try:
        conn = sqlite3.connect('bot.sqlite')

        cursor = conn.cursor()

        query = '''
            CREATE TABLE IF NOT EXISTS feedback(
                id INTEGER PRIMARY KEY, 
                key TEXT UNIQUE,
                value TEXT
            )
        '''

        cursor.execute(query)
    except Exception as e:
        raise e
    finally:
        conn.commit()
        conn.close()


def get_feedback_response(request):
    try:
        conn = sqlite3.connect('bot.sqlite')
        cursor = conn.cursor()

        query = '''
            SELECT key, value
            FROM feedback
            WHERE key = ?
        '''

        cursor.execute(query, (request,))
        value = cursor.fetchone()
    except:
        return None
    finally:
        conn.commit()
        conn.close()
    
    return value

def set_feedback_response(request, response):
    try:
        conn = sqlite3.connect('bot.sqlite')
        cursor = conn.cursor()

        query = '''
            INSERT OR REPLACE INTO feedback (key, value)
            VALUES(?,?);
        '''

        cursor.execute(query, (request, response))
    except Exception as e:
        raise e
    finally:
        conn.commit()
        conn.close()

def get_feedback(request):
    text = input("\nIs the response correct(Yes/No): ")

    if 'yes' in text.lower():
        return True
    elif 'no' in text.lower():
        correct_response = input('please input the correct one: ')
        set_feedback_response(request, correct_response)
        print('Responses added to bot!')
        return True
    else:
        return get_feedback()

def chat():
    print("Start talking with the bot (type 'quit' or 'q' to stop)!")
    while True:
        print('\x1b[6;30;42m' + 'You:' + '\x1b[0m', end=' ')
        inp = input()
        if inp.lower() == "quit" or inp.lower() == "q":
            print('Bye Bye!')
            break
        elif inp is None:
            continue

        previous_response = get_feedback_response(inp.lower())

        if(previous_response is None):
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            print(results[results_index])

            if results[results_index] > 0.70:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                response = random.choice(responses)
            else:
                response = duckduckgo_response(inp)
        else:
            response = previous_response[1]

        print('\x1b[6;30;44m' + 'Bot:' + '\x1b[0m ', response)
        
        get_feedback(inp.lower())

create_table()
chat()
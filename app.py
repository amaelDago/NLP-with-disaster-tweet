import numpy as np
import random
import json
from flask import Flask, render_template, request
from utils import classifier, tweet_to_tensor

# Instanciate model
#model = classifier()

a_file = open("vocab.json", "r")
word2ind = a_file.read()

unknown = "__unk__"


# Upload wieghts
#model.init_from_file('model.pkl.gz')

# Define application
app = Flask(__name__)

# Deine route for homepage
@app.route("/")

def index() : 
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict() : 

	feature = [str(x) for x in request.form.values()]

	input = np.array(tweet_to_tensor(feature[0], unknown, word2ind))

	# Classification using model
	pred = model(input)
	return render_template('index.html', prediction_text= 'Tweet \"{}\" concerns a {} event with a probability of {}%'.format(feature[0], bool(pred), round(prob*100,2)))

if __name__ == "__main__" : 
    app.run(debug = True)
	

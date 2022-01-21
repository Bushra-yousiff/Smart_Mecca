from flask import Flask, render_template, redirect, request
import warnings
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model
import caption_mecca
import matplotlib.pyplot as plt
import pickle
import numpy as np



warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")
@app.route('/',methods=['POST'])
def marks():
	if request.method == 'POST':
		f= request.files['userfile']
		path = './static/{}'.format(f.filename)
		f.save(path)
		caption_2=caption_mecca.ar_speech(path)
		print(caption_2)

		result_dict = { 
		'image': path,
		'caption': caption_2}

	return render_template("index.html",your_result=result_dict)
if __name__ == '__main__':
	app.run(debug = True)
	#app.run(host='0.0.0.0')

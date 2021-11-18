from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
import formatting

app = Flask(__name__)

# Load the model
rec_dict = pickle.load(open('./trg/prod_rec_dict.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
	data = request.get_json(force=True)['feature']
	age_grp = formatting.age_grp(data[0])
	gender = data[1]
	region = 'region_'+str(data[2])
	key = (age_grp, gender, region)
	# predict = rec_dict[key]
	# return jsonify(predict)
	return jsonify(key)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
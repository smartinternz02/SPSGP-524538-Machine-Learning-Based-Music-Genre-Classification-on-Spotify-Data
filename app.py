# Importing the necessary dependencies
from flask import Flask, request, render_template
import numpy as np
import pickle

# Loading the model
app = Flask(__name__) # initializing a flask app
with open('D:\Applied Data Science\Flask\model.plk','rb') as f:
    model = pickle.load(f)
with open('D:\Applied Data Science\Flask\scalar.plk','rb') as f:
    sc = pickle.load(f)

# Loading the home page
@app.route('/') # route to display home page
def home():
    return render_template('home.html') # rendering the home page

# Loading the Music Genre Prediction page
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('index.html')

@app.route('/home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # reading the inputs by the user
    input_features = [float(x) for x in request.form.values()]
    x = [np.array(input_features)]
    x = sc.transform(x)
    
    # prediction using the loaded model
    prediction = model.predict(x)
    labels = ['Dark Trap', 'Underground Rap', 'Trap Metal', 'Emo', 'Rap', 'RnB',
       'Pop', 'Hiphop', 'Tech House', 'Techno', 'Trance', 'Psytrance',
       'Trap', 'DnB', 'Hardstyle']
    
    # showing the prediction result
    return render_template('result.html', prediction = labels[prediction[0]])

# running the app
if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)



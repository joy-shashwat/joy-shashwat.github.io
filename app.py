#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('dashboard.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('DS/ds_project1.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
#######################
# DATA SCIENCE PROJECTS
#######################
@app.route('/datascience')
def datascience():
    return render_template('DS/datascience.html')

@app.route('/ds_project1')
def ds_project1():
    return render_template('DS/ds_project1.html')

@app.route('/ds_project2')
def ds_project2():
    return render_template('DS/ds_project2.html')

@app.route('/ds_project3')
def ds_project3():
    return render_template('DS/ds_project3.html')

######################################
# NATURAL LANGUAGE PROCESSING PROJECTS
######################################
@app.route('/nlp')
def nlp():
    return render_template('NLP/nlp.html')

@app.route('/nlp_project1')
def nlp_project1():
    return render_template('NLP/nlp_project1.html')

@app.route('/nlp_project2')
def nlp_project2():
    return render_template('NLP/nlp_project2.html')

@app.route('/nlp_project3')
def nlp_project3():
    return render_template('NLP/nlp_project3.html')

##########################
# COMPUTER VISION PROJECTS
##########################
@app.route('/computervision')
def computervision():
    return render_template('CV/computervision.html')

@app.route('/cv_project1')
def cv_project1():
    return render_template('CV/cv_project1.html')

@app.route('/cv_project2')
def cv_project2():
    return render_template('CV/cv_project2.html')

@app.route('/cv_project3')
def cv_project3():
    return render_template('CV/cv_project3.html')

#####################################
#####################################

if __name__ == "__main__":
    app.run(debug=True)
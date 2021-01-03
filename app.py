#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle

# DS libs
import os
import joblib

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# initialize Models pkg
# DataScience ML Packages
news_vectorizer = open(os.path.join("static/models/final_news_cv_vectorizer.pkl"),"rb")
news_cv = joblib.load(news_vectorizer)

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key 

#default page of our web-app
@app.route('/')
def home():
    return render_template('dashboard.html')

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

#To use the predict button in our web-app
@app.route('/ds_predict1',methods=['POST'])
def ds_predict1():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('DS/ds_project1.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

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


@app.route('/nlp_predict1',methods=['GET', 'POST'])
def nlp_predict1():
     if request.method == 'POST':
        rawtext = request.form['rawtext']
        modelchoice = request.form['modelchoice']
        vectorized_text = news_cv.transform([rawtext]).toarray()
        
        if modelchoice == 'nb':
            news_nb_model = open(os.path.join("static/models/newsclassifier_NB_model.pkl"),'rb')
            news_clf = joblib.load(news_nb_model)
        elif modelchoice == 'logit':
            news_nb_model = open(os.path.join("static/models/newsclassifier_Logit_model.pkl"),'rb')
            news_clf = joblib.load(news_nb_model)
        elif modelchoice == 'rf':
            news_nb_model = open(os.path.join("static/models/newsclassifier_RFOREST_model.pkl"),'rb')
            news_clf = joblib.load(news_nb_model)

        #Prediction
        prediction_labels = {"business":0,"tech":1,"sport":2,"health":3,"politics":4,"entertainment":5}
        prediction = news_clf.predict(vectorized_text)
        final_result = get_keys(prediction,prediction_labels)

        return render_template('NLP/nlp_project1.html',rawtext = rawtext.upper(),final_result=final_result)


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
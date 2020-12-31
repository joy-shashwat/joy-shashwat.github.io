#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
import pickle

# DS libs
import os
import joblib
#jigar DS
from flask_cors import cross_origin
import sklearn
import pandas as pd



#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#flight price prediction
model = pickle.load(open("static/models_ds/plane_rf.pkl", "rb"))

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

@app.route("/ds_predict2", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # print("Journey Date : ",Journey_day, Journey_month)

        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # print("Departure : ",Dep_hour, Dep_min)

        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # print("Arrival : ", Arrival_hour, Arrival_min)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # print("Duration : ", dur_hour, dur_min)

        # Total Stops
        Total_stops = int(request.form["stops"])
        # print(Total_stops)

        # Airline
        # AIR ASIA = 0 (not in column)
        airline=request.form['airline']
        #initialization and putting them all as Zero
        Jet_Airways = 0
        IndiGo = 0
        Air_India = 0
        Multiple_carriers = 0
        SpiceJet = 0
        Vistara = 0
        GoAir = 0
        Multiple_carriers_Premium_economy = 0
        Jet_Airways_Business = 0
        Vistara_Premium_economy = 0
        Trujet = 0 

        if(airline=='Jet Airways'):
            Jet_Airways = 1
        elif (airline=='IndiGo'):
            IndiGo = 1
        elif (airline=='Air India'):
            Air_India = 1
        elif (airline=='Multiple carriers'):
            Multiple_carriers = 1
        elif (airline=='SpiceJet'):
            SpiceJet = 1
        elif (airline=='Vistara'):
            Vistara = 1
        elif (airline=='GoAir'):
            GoAir = 1
        elif (airline=='Multiple carriers Premium economy'):
            Multiple_carriers_Premium_economy = 1
        elif (airline=='Jet Airways Business'):
            Jet_Airways_Business = 1
        elif (airline=='Vistara Premium economy'):
            Vistara_Premium_economy = 1
        elif (airline=='Trujet'):
            Trujet = 1
        else:
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        # Source
        #initialisation to Zero
        s_Delhi = 0
        s_Kolkata = 0
        s_Mumbai = 0
        s_Chennai = 0
        # Banglore = 0 (not in column)
        Source = request.form["Source"]
        if (Source == 'Delhi'):
            s_Delhi = 1
        elif (Source == 'Kolkata'):
            s_Kolkata = 1
        elif (Source == 'Mumbai'):
            s_Mumbai = 1
        elif (Source == 'Chennai'):
            s_Chennai = 1
        else:
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0

        # print(s_Delhi,
        #     s_Kolkata,
        #     s_Mumbai,
        #     s_Chennai)

        # Destination
        # Banglore = 0 (not in column)
        Source = request.form["Destination"]
        if (Source == 'Cochin'):
            d_Cochin = 1
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        
        elif (Source == 'Delhi'):
            d_Cochin = 0
            d_Delhi = 1
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0

        elif (Source == 'New_Delhi'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 1
            d_Hyderabad = 0
            d_Kolkata = 0

        elif (Source == 'Hyderabad'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 1
            d_Kolkata = 0

        elif (Source == 'Kolkata'):
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 1

        else:
            d_Cochin = 0
            d_Delhi = 0
            d_New_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
        
        prediction=model.predict([[
            Total_stops,
            Journey_day,
            Journey_month,
            Dep_hour,
            Dep_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
            Air_India,
            GoAir,
            IndiGo,
            Jet_Airways,
            Jet_Airways_Business,
            Multiple_carriers,
            Multiple_carriers_Premium_economy,
            SpiceJet,
            Trujet,
            Vistara,
            Vistara_Premium_economy,
            s_Chennai,
            s_Delhi,
            s_Kolkata,
            s_Mumbai,
            d_Cochin,
            d_Delhi,
            d_Hyderabad,
            d_Kolkata,
            d_New_Delhi
        ]])

        output=round(prediction[0],2)

    return render_template('DS/ds_project2.html',prediction_text="Your Flight price is Rs. {}".format(output))

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
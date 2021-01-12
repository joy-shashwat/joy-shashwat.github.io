#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

# DS libs
import os
import joblib

#import Air Fare prediction
import pandas as pd

# CV imports
import os 
import sys
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import face_recognition
import cv2
import numpy as np

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('static/models_ds1/model.pkl', 'rb'))

# initialize Models pkg
# DataScience ML Packages
news_vectorizer = open(os.path.join("static/models_nlp1/final_news_cv_vectorizer.pkl"),"rb")
news_cv = joblib.load(news_vectorizer)


#CV packages
UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = set(['txt','mp4'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

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
def ds_predict2():
    if request.method == "POST":
        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
        Journey_month = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").month)
        # Departure
        Dep_hour = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").hour)
        Dep_min = int(pd.to_datetime(date_dep, format ="%Y-%m-%dT%H:%M").minute)
        # Arrival
        date_arr = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").hour)
        Arrival_min = int(pd.to_datetime(date_arr, format ="%Y-%m-%dT%H:%M").minute)
        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)
        # Total Stops
        Total_stops = int(request.form["stops"])
        # Airline
        # AIR ASIA = 0 (not in column)
        airline=request.form['airline']
        if(airline=='Jet Airways'):
            Jet_Airways = 1
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

        elif (airline=='IndiGo'):
            Jet_Airways = 0
            IndiGo = 1
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 

        elif (airline=='Air India'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 1
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='Multiple carriers'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 1
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='SpiceJet'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 1
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0 
            
        elif (airline=='Vistara'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 1
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='GoAir'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 1
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Multiple carriers Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 1
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Jet Airways Business'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 1
            Vistara_Premium_economy = 0
            Trujet = 0

        elif (airline=='Vistara Premium economy'):
            Jet_Airways = 0
            IndiGo = 0
            Air_India = 0
            Multiple_carriers = 0
            SpiceJet = 0
            Vistara = 0
            GoAir = 0
            Multiple_carriers_Premium_economy = 0
            Jet_Airways_Business = 0
            Vistara_Premium_economy = 1
            Trujet = 0
            
        elif (airline=='Trujet'):
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
        # Banglore = 0 (not in column)
        Source = request.form["Source"]
        if (Source == 'Delhi'):
            s_Delhi = 1
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0

        elif (Source == 'Kolkata'):
            s_Delhi = 0
            s_Kolkata = 1
            s_Mumbai = 0
            s_Chennai = 0

        elif (Source == 'Mumbai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 1
            s_Chennai = 0

        elif (Source == 'Chennai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 1

        else:
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0

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

        return render_template('DS/ds_predict2.html',prediction_text="Your Flight price is Rs. {}".format(output))

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
            news_nb_model = open(os.path.join("static/models_nlp1/newsclassifier_NB_model.pkl"),'rb')
            news_clf = joblib.load(news_nb_model)
        elif modelchoice == 'logit':
            news_nb_model = open(os.path.join("static/models_nlp1/newsclassifier_Logit_model.pkl"),'rb')
            news_clf = joblib.load(news_nb_model)
        elif modelchoice == 'rf':
            news_nb_model = open(os.path.join("static/models_nlp1/newsclassifier_RFOREST_model.pkl"),'rb')
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

#@app.route('/cv_project1')
#def cv_project1():
#    return render_template('CV/cv_project1.html')

@app.route("/cv_project1", methods=['GET', 'POST'])
def cv_project1():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            NEWPATH=videotest(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('CV/cv_project1.html')
    return render_template('CV/cv_test.html')

def videotest(filename):
    video_capture = cv2.VideoCapture(filename)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    harmesh = face_recognition.load_image_file("static/models_cv1/harmesh.jpg")
    hfencoding = face_recognition.face_encodings(harmesh)[0]

    prateek = face_recognition.load_image_file("static/models_cv1/prateek.jpeg")
    pfencoding = face_recognition.face_encodings(prateek)[0]

   
    known_face_encodings = [
        hfencoding,
        pfencoding
    ]
    known_face_names = [
        "Harmesh",
        "Prateek"
    ]

    width  = int(video_capture.get(3)) # float
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    PATH = 'static/demo.webm'
    out = cv2.VideoWriter(PATH,fourcc, fps, (width,height))
    for i in range(1,length-1):
        
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 10), (right, bottom + 10 ), (10, 10, 10), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 2, bottom), font, 0.4, (255, 255, 255), 1)

        print()
        sys.stdout.write(f"writing...{int((i/length)*100)+1}%")
        sys.stdout.flush()
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    return PATH


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
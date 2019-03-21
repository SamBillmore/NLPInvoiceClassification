import os
import flask
import operator
from werkzeug.utils import secure_filename
from flask import Flask,render_template,flash, request, redirect, url_for,send_from_directory,send_file,session

#-------- Config Variables -----------#
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['csv'])
RESULT_FOLDER = './results/'

#-------- app creation -----------#

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'woodsecretkey'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

#-------- MODEL GOES HERE -----------#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from preprocessing_classes import FeatureExtractor, ToNumeric, TextPreprocessor, Dummifier

with open('./model.pkl', 'rb') as picklefile:
    model = pickle.load(picklefile)

with open('./data_processing.pkl', 'rb') as picklefile:
    pre_process = pickle.load(picklefile)

#-------- Functions Definition -----------#

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#-------- ROUTES GO HERE -----------#

@app.route('/')
@app.route('/page')
def page():
    return render_template('index1.html')

@app.route('/manualupload',methods=['GET','POST'])
def manualupload():
    return render_template('manualupload.html')

@app.route('/fileupload' ,methods=['GET','POST'])
def fileupload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Please select a file and submit')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File should be of type csv')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('fileres',
                                    filename=filename))
    return render_template('fileupload.html')

@app.route('/dbupload' ,methods=['GET','POST'])
def dbupload():
    return render_template('dbupload.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html') ,404

@app.route('/res',methods=['GET','POST'])
def res():
    if flask.request.method == 'POST':
        inputs = flask.request.form

        bus_unit = inputs['BusinessUnit']
        supp_name = inputs['SupplierName']
        supp_grp = inputs['SupplierGroup']
        inv_desc = inputs['InvoiceDescription']
        inv_amt = inputs['InvoiceAmount']
        inv_curr = inputs['InvoiceCurrency']
        usd_amt = inputs['USDAmount']
        prj_org = inputs['ProjectOwningOrg']
        datasource = inputs['DataSource']
        legacy = inputs['Legacy']
        year = inputs['Year']
        leak_id = inputs['Leakageflag']
        leak_grp = inputs['LeakageGroup']
        inter_co = inputs['Intercompanyflag']
        am_flag = inputs['AmericasFlag']

        item = pd.DataFrame(np.array([bus_unit, supp_name, supp_grp, inv_desc, inv_amt, inv_curr, usd_amt, prj_org, datasource, legacy, year, leak_id, leak_grp, inter_co, am_flag]).reshape(1, -1),
                    columns=['Business Unit', 'Supplier Name', 'Supplier_Group', 'Invoice Desc', 'Invoice_Amt', 'Invoice Currency', 'USD_Amt', 'Project Owning Org', 'datasource', 'Legacy', 'Year',
                    'Leakage_Identifier', 'Leakage_Group', 'Intercompany_Flag', 'Americas_Flag'])
        item = pre_process.transform(item)
        score = model.predict_proba(item)
        prediction = model.predict(item)[0]
        certainty = max(score[0])

        result_pred={'PREDICTED CATEGORY': prediction,
                    'CERTAINTY': certainty * 100 }

        result = {  'CHEMICALS & CATALYSTS': round(score[0,0]*100,2),
                    'CONSTRUCTION MATERIALS': round(score[0,1]*100,2),
                    'CONSTRUCTION SERVICES': round(score[0,2]*100,2),
                    'CORPORATE SERVICES': round(score[0,3]*100,2),
                    'ELECTRICAL': round(score[0,4]*100,2),
                    'ENGINEERING SERVICES': round(score[0,5]*100,2),
                    'ENVIRONMENTAL SERVICES': round(score[0,6]*100,2),
                    'FABRIC MAINTENANCE': round(score[0,7]*100,2),
                    'FACILITY MANAGEMENT': round(score[0,8]*100,2),
                    'FLEET AND RENTAL EQUIPMENT': round(score[0,9]*100,2),
                    'INFORMATION TECHNOLOGY': round(score[0,10]*100,2),
                    'INSTRUMENTATION & AUTOMATION': round(score[0,11]*100,2),
                    'LOGISTICS': round(score[0,12]*100,2),
                    'MECHANICAL EQUIPMENT': round(score[0,13]*100,2),
                    'PIPING': round(score[0,14]*100,2),
                    'PROJECT INDIRECT': round(score[0,15]*100,2),
                    'STRUCTURAL STEEL': round(score[0,16]*100,2),
                    'VALVES': round(score[0,17]*100,2)
                        }
        tuple_array=[]
        i=0
        sorted_d = sorted(result.items(),key=operator.itemgetter(1),reverse=True)
        for val in sorted_d:
            i=i+1
            tuple_array.append((i,val[0],val[1]))
        return render_template('res.html',result=result,tuple_array=tuple_array,result_pred=result_pred)

    return render_template('res.html')


@app.route('/fileres/<filename>',methods=['GET','POST'])
def fileres(filename):
    fn=send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    input_df=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), skiprows=1, names=['Business Unit', 'Supplier Name', 'Supplier_Group', 'Invoice Desc', 'Invoice_Amt', 'Invoice Currency', 'USD_Amt', 'Project Owning Org', 'datasource', 'Legacy', 'Year',
              'Leakage_Identifier', 'Leakage_Group', 'Intercompany_Flag', 'Americas_Flag'])
    rows=input_df.shape[0]
    input_df_cp=input_df.copy()
    input_df = pre_process.transform(input_df)
    score = model.predict_proba(input_df)
    score_df=pd.DataFrame(score,columns=model.classes_)
    prediction = model.predict(input_df)
    prediction_df=pd.DataFrame(prediction,columns=['predicted_class'])
    prediction_df['predicted_proba']=score_df.max(axis=1)
    final_df=pd.concat([input_df_cp,prediction_df, score_df,], axis=1)
    fn_res=filename.split('.')[0] + '_result.csv'
    final_df.to_csv(os.path.join(app.config['RESULT_FOLDER'], fn_res), encoding='utf-8', index=False)
    session['fn_res']=fn_res
    return render_template('fileres.html',rows=rows)


@app.route('/dbres',methods=['GET','POST'])
def dbres():
    if flask.request.method == 'POST':
        inputs = flask.request.form
        return render_template('dbres.html')

    return render_template('dbres.html')

@app.route('/download/')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    #path = "C:/Users/hy846tk/Documents/Wood/Flask_app/results/Test_data_for_flask_app.csv"
    return send_file(os.path.join(app.config['RESULT_FOLDER'], session['fn_res']), as_attachment=True)

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 5000

    app.run(HOST, PORT,debug=True)

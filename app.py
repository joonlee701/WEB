from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.models import load_model

app=Flask(__name__)

model_iris_deep=None
model_iris_lr=None
model_iris_svm=None
model_iris_dt=None
def load_iris():
    global model_iris_deep,model_iris_lr,model_iris_svm,model_iris_dt,
    model_iris_lr = joblib.load('model/iris_lr.pkl')
    model_iris_svm= joblib.load('model/iris_svm.pkl')
    model_iris_dt = joblib.load('model/iris_dt.pkl')
    model_iris_deep=joblib.load('model/iris_deep.hdf5')


@app.route('/')
def index():
    return render_template('01_home.html')

@app.route('/typo')
def typo():  
#유니크하다? -> local host 5000/ typo면 typo로 가라! 그렇게 되면 중복될 염려가 없음. 
    return render_template('03_typography.html')

@app.route('/project')
def project():
    return render_template('17_templates.html')

@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/iris', methods=['GET',"POST"])
def iris():
    if request.method == 'GET':
        return render_template('12_form-iris.html')
    else:
        slen1=float(request.form['slen']) *2
        plen1=float(request.form['plen'])*2
        pwid1=float(request.form['pwid'])*2
        species1=int(request.form['species'])*2
        comment1=request.form['comment']*2
        return render_template('12_iris-result.html',
            slen=slen,plen=plen,pwid=pwid,species=species,comment=comment)



if __name__ == '__main__':
    load_iris()
    app.run(debug=True)
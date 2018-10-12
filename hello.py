from flask import Flask,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


app = Flask(__name__)

@app.route('/')
def index():

	train=pd.read_csv('D:/Assignment/Python/Sample DataSets/pima-indians-diabetes.csv')
	test=pd.read_csv('D:/Assignment/Python/Sample DataSets/test.csv')
	x_train = train.values
	y_train = train["Target"].values
	x_test=test.values
	rf = RandomForestClassifier(n_estimators=1000)
	rf.fit(x_train, y_train)
	status = rf.predict_proba(x_train)
	fpr, tpr, _ = roc_curve(y_train, status[:,1])
	roc_auc = auc(fpr, tpr)
	print(roc_auc)

	return render_template('hello.html', name=roc_auc)
	
	


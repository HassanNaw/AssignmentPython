from flask import Flask,render_template
import numpy as np
import pandas as pd
import array as arr
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():

	complete=pd.read_csv('D:/Assignment/Python/Sample DataSets/Pima_India/pima_indians_diabetesComplete.csv')
	complete_data = pd.DataFrame()
	complete_data = complete
	
	#80 % for Train :
	train_data = complete_data.iloc[:600,:]
	
	ClassLabel = train_data.values [:,8]
	RemainingCols = train_data.values [:,:7]
	RC_Train = pd.DataFrame(RemainingCols)
	
	Classifer_ = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
	            max_depth=None, max_features='auto', max_leaf_nodes=None,
	            min_impurity_decrease=0.0, min_impurity_split=None,
	            min_samples_leaf=1, min_samples_split=2,
	            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
	            oob_score=False, random_state=None, verbose=0,
	            warm_start=False)
	
	Classifer_.fit(RC_Train,ClassLabel)
	
	TrainScoreTest= Classifer_.predict(RC_Train)
	#print ("RandomForestClassifier Classifier Accuracy is ",accuracy_score (ClassLabel,TrainScoreTest)*100)
	confusion_matrix(ClassLabel, TrainScoreTest)
	
	#saving model\n",
	filename = 'finalized_Classifier.sav'
	joblib.dump(Classifer_,filename)
	
	#loading model
	loaded_model = joblib.load(filename)
	result = loaded_model.score(RemainingCols,ClassLabel)
	#print(result)
	
	#Scoring Data Split :
	test_data = complete_data.iloc[600:,:]
	
	#SCORING
	ClassLabel = test_data.values [:,8]
	RemainingCols = test_data.values [:,:7]
	RC_Score = pd.DataFrame(RemainingCols)
	ClassLabelScore = pd.DataFrame(ClassLabel)
	Scoring = loaded_model.predict(RemainingCols)
	#print ("RandomForestClassifier Classifier Accuracy is ", accuracy_score(ClassLabel,Scoring)*100)
	Final_Accuracy = accuracy_score(ClassLabel,Scoring)*100
	#output =confusion_matrix(ClassLabel, Scoring)
	print(Final_Accuracy)
	
	numbers =[0]
	
	numbers[0]=Final_Accuracy
	
	return render_template('Accuracy.html', name=numbers[0])



# import pickle
# import sys

import joblib
# import numpy as np


# Load the model
model = joblib.load('xgb_classifier.joblib')


# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
# input = [885.157845,853.763730,9.063146,-0.000179,2.143342, 2661.894136,72.203287]

str = '885.157845,853.763730,9.063146,-0.000179,2.143342,2661.894136,72.203287'


def Convert(string):
    li = [float(x) for x in string.split(",")]
    return li


input = Convert(str)
# input = Convert(sys.argv[1])
prediction = model.predict([input])
print("hello")
print(prediction)

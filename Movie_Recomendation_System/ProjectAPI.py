import logging
import joblib
import sys
import pandas as pd
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from datetime import datetime
import pickle
from ML_project_Final import recc
from ML_project_Final import recc2

app = FastAPI()

# class movieBased(BaseModel):
#     movie_name : str
        
# class userBased(BaseModel):
#     userID : str

pickle_in1 = open('recc.pkl', 'rb')
a = pickle.load(pickle_in1)

pickle_in2 = open('recc2.pkl', 'rb')
a2 = pickle.load(pickle_in2)

@app.get('/status/')
def HealthCheck():
    return 'HI'

@app.get("/api/recommendations/")
def predict1(movie_name):
    return(a.get_recommendations(movie_name, 10))

@app.get('/api/recommendations2/')
def predict2(userID):
    # userID = userBased.userID
    return(a2.get_user_recommendations(userID , 2))
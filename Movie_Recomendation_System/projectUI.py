import streamlit as st
from pydantic import BaseModel
import requests
import io
import pickle
import pandas as pd
# url = 'http://localhost:8080/docs#/'
st.title('Movie Recommendation')
st.header('Recommended for you')
userID = st.text_input('Enter UserID')
b1 = st.button('Click')
if(b1):
    result = requests.get(f'http://127.0.0.1:8000/api/recommendations2/?userID={userID}')
    tempList2 = (result.content).decode('utf-8')
    tempList2 = tempList2.split('"')
#     tempList2 = result.content.decode('utf-8')
#     r = tempList2.json()
#     st.write(r)
    df2 = pd.DataFrame(tempList2)
    st.table(df2)
#     st.write((result.content).decode('utf-8'))
    
st.header('Find Similar Movies')
movieName = st.text_input('Enter Movie Name')
b2 = st.button('Here')
if(b2):
    movie_name = movieName.replace(' ', '%20')
    movie_name = movieName.replace('(', '%28')
    movie_name = movieName.replace(')', '%29')
    result2 = requests.get(f'http://127.0.0.1:8000/api/recommendations/?movie_name={movie_name}')
#     st.write((result2.content).decode('utf-8'))
    tempList = (result2.content).decode('utf-8')
    tempList = tempList.split('"')
    df = pd.DataFrame(tempList)
    st.table(df)
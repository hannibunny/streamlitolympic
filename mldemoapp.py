
### Part 1: Import necessary packages #######################################################
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

weights={"lin":[-1,0.5],
   "pol":[0.1,0.3,0.1,-0.03],
   "log":[0.5,1.0],
   "sin":[1,0.2,-0.5,0.7]}

def visualize_data(w,lower=0.1,upper=5,res=0.01,k=0.1):
    st.markdown(f"**4 different sets of Training Data**")
    np.random.seed(123)
    x=np.arange(lower,upper,res)
    ###### 1. Linear
    w0=w["lin"][0]
    w1=w["lin"][1]
    y1 = w0+w1*x+k*np.random.randn(len(x))
    ####### 2. Polynomial 
    w0=w["pol"][0]
    w1=w["pol"][1]
    w2=w["pol"][2]
    w3=w["pol"][3]
    y2= w0+w1*x+w2*x**2+w3*x**3+k*np.random.randn(len(x))
    ### 3. logarithmic
    w0=w["log"][0]
    w1=w["log"][1]
    y3 = w0+w1*np.log10(x+1)+k*np.random.randn(len(x)) 
    ### 4. sinusiod
    w0=w["sin"][0]
    w1=w["sin"][1]
    w2=w["sin"][2]
    w3=w["sin"][3]
    y4=w0+w1*x+w2*np.sin(3*x)+w3*3*np.cos(6*x)+2*k*np.random.randn(len(x)) 

    fig = make_subplots(rows=2, cols=2, start_cell="top-left")#,column_titles=["x","x"],row_titles=["y","y"])
    

    fig.add_trace(go.Scatter(x=x, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot"),name="Data 1"),
                  row=1, col=1)
    #fig.add_trace(go.Scatter(x=x, y=y1r, line = dict(color = 'orange'),name="Linear"),
    #              row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2,mode='markers', marker = dict(color = 'green',symbol="circle-dot"),name="Data 2"),row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y3,mode='markers', marker = dict(color = 'brown',symbol="circle-dot"),name="Data 3"),row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y4,mode='markers', marker = dict(color = 'orange',symbol="circle-dot"),name="Data 4"),row=2, col=2)
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2)
    
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_yaxes(title_text="y", row=2, col=1)
    fig.update_yaxes(title_text="y", row=2, col=2)
    fig.update_layout(height=700,xaxis_title="x",title="Training Data")

    st.plotly_chart(fig)

def visualize_models(w,lower=0.1,upper=5,res=0.01,k=0.1):
    st.markdown(f"**Models (functions) which have been learned from the given data**")
    np.random.seed(123)
    x=np.arange(lower,upper,res)
    xm=np.arange(lower-0.5,upper+0.5,res)
    ###### 1. Linear
    w0=w["lin"][0]
    w1=w["lin"][1]
    m1= w0+w1*x
    y1 = m1+k*np.random.randn(len(x))
    m1m=w0+w1*xm
    ####### 2. Polynomial 
    w0=w["pol"][0]
    w1=w["pol"][1]
    w2=w["pol"][2]
    w3=w["pol"][3]
    m2=w0+w1*x+w2*x**2+w3*x**3
    y2= m2+k*np.random.randn(len(x))
    m2m=w0+w1*xm+w2*xm**2+w3*xm**3
    ### 3. logarithmic
    w0=w["log"][0]
    w1=w["log"][1]
    m3=w0+w1*np.log10(x+1)
    y3 = m3+k*np.random.randn(len(x))
    m3m=w0+w1*np.log10(xm+1)
    ### 4. sinusiod
    w0=w["sin"][0]
    w1=w["sin"][1]
    w2=w["sin"][2]
    w3=w["sin"][3]
    m4=w0+w1*x+w2*np.sin(3*x)+w3*3*np.cos(6*x)+2
    y4=m4+2*k*np.random.randn(len(x))
    m4m=w0+w1*xm+w2*np.sin(3*xm)+w3*3*np.cos(6*xm)+2

    fig = make_subplots(rows=2, cols=2, start_cell="top-left")#,column_titles=["x","x"],row_titles=["y","y"])
    

    fig.add_trace(go.Scatter(x=x, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot"),name="Data 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xm, y=m1m, line = dict(color = 'red',width=3),name="Learned Model"),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=y2,mode='markers', marker = dict(color = 'green',symbol="circle-dot"),name="Data 2"),row=1, col=2)
    fig.add_trace(go.Scatter(x=xm, y=m2m, line = dict(color = 'red',width=3),name="Learned Model"),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=x, y=y3,mode='markers', marker = dict(color = 'brown',symbol="circle-dot"),name="Data3"),row=2, col=1)
    fig.add_trace(go.Scatter(x=xm, y=m3m, line = dict(color = 'red',width=3),name="Learned Model"),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=y4,mode='markers', marker = dict(color = 'orange',symbol="circle-dot"),name="Data4"),row=2, col=2)
    fig.add_trace(go.Scatter(x=xm, y=m4m, line = dict(color = 'red',width=3),name="Learned Model"),
                  row=2, col=2)
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2)
    
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_yaxes(title_text="y", row=2, col=1)
    fig.update_yaxes(title_text="y", row=2, col=2)
    fig.update_layout(height=700,xaxis_title="x",title="Training Data and Learned Models")

    st.plotly_chart(fig)

###### Main Program #######
st.header("Demonstration of Machine Learning")

show = st.radio(
    "Select One",
    ["Show Data", "Show trained models"],
    index=0,
)

if show == "Show Data":
    visualize_data(w=weights)
else:
    visualize_models(w=weights)

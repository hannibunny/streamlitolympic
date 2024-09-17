
### Part 1: Import necessary packages #######################################################
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def visualize_data(w,lower=0.1,upper=5,res=0.01,k=0.15):
    st.markdown(f"**4 different sets of Training Data**:") 
    st.html("Here, in each set the input is 1-dimensional (value of x) and the target-output is y. From the given Input/output-pairs (x,y) the algorithm must learn a function y=f(x)")
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
    y4=w0+w1*x+w2*np.sin(3*x)+w3*3*np.cos(6*x)+3*k*np.random.randn(len(x)) 

    fig = make_subplots(rows=2, cols=2, start_cell="top-left")#,column_titles=["x","x"],row_titles=["y","y"])
    

    fig.add_trace(go.Scatter(x=x, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot",opacity=0.5),name="Data 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2,mode='markers', marker = dict(color = 'green',symbol="circle-dot",opacity=0.5),name="Data 2"),row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y3,mode='markers', marker = dict(color = 'brown',symbol="circle-dot",opacity=0.5),name="Data 3"),row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y4,mode='markers', marker = dict(color = 'orange',symbol="circle-dot",opacity=0.5),name="Data 4"),row=2, col=2)
    
    fig.update_xaxes(title_text="x", range=[lower-0.5,upper+0.5],row=1, col=1)
    fig.update_xaxes(title_text="x", range=[lower-0.5,upper+0.5],row=1, col=2)
    fig.update_xaxes(title_text="x", range=[lower-0.5,upper+0.5],row=2, col=1)
    fig.update_xaxes(title_text="x", range=[lower-0.5,upper+0.5],row=2, col=2)
    
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
    y4=m4+3*k*np.random.randn(len(x))
    m4m=w0+w1*xm+w2*np.sin(3*xm)+w3*3*np.cos(6*xm)+2

    fig = make_subplots(rows=2, cols=2, start_cell="top-left",
                        subplot_titles=('Learned: y=-1+0.5x', 
                                        "Learned: y=0.1 +0.3x +0.1x<sup>2</sup> -0.03x<sup>3</sup>", 
                                        "Learned: y=0.5+1log<sub>10</sub>(x)",
                                        "Learned: y= 1 +0.2x +0.6sin(3x)-0.7cos(6x)"))
    

    fig.add_trace(go.Scatter(x=x, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot",opacity=0.5),name="Data 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xm, y=m1m, line = dict(color = 'red',width=3),name="Learned"),
                  row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=y2,mode='markers', marker = dict(color = 'green',symbol="circle-dot",opacity=0.5),name="Data 2"),row=1, col=2)
    fig.add_trace(go.Scatter(x=xm, y=m2m, line = dict(color = 'red',width=3),name="Learned"),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=x, y=y3,mode='markers', marker = dict(color = 'brown',symbol="circle-dot",opacity=0.5),name="Data3"),row=2, col=1)
    fig.add_trace(go.Scatter(x=xm, y=m3m, line = dict(color = 'red',width=3),name="Learned"),
                  row=2, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=y4,mode='markers', marker = dict(color = 'orange',symbol="circle-dot",opacity=0.5),name="Data4"),row=2, col=2)
    fig.add_trace(go.Scatter(x=xm, y=m4m, line = dict(color = 'red',width=3),name="Learned"),
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



def is_inside_ellipse(x, y,x0,y0,a,b):
    return ((x - x0)**2 / a**2 + (y - y0)**2 / b**2) < 1


def visualize_data_class(w,lower=0,upper=5,n_points=80):
    n_partition = n_points // 2
    st.markdown(f"**4 different sets of Training Data**:") 
    st.html("Here, in each set the input is 2-dimensional (x<sub>1</sub>,x<sub>2</sub>) and the target-output y is the label which is indicated by the marker color. From the given input/output-pairs the algorithm must learn a function, which maps each input to a class: y=f(x<sub>1</sub>,x<sub>2</sub>)")
    np.random.seed(123)

    # Lineare Trennungsfunktion: y = 0.5x + 1
    # Partition 1: Punkte unter der Linie
    x1 = np.random.uniform(0, 5, n_partition * 2)
    y1 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y1 < w["lin"][1] * x1 + w["lin"][0]  # Bedingung für Partition 1
    x1, y1 = x1[mask1][:n_partition], y1[mask1][:n_partition]  # Beschränke auf n_points /2 Punkte
    
    # Partition 2: Punkte über der Linie
    x2 = np.random.uniform(0, 5, n_partition * 2)
    y2 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y2 >= w["lin"][1] * x2 + w["lin"][0]  # Bedingung für Partition 2
    x2, y2 = x2[mask2][:n_partition], y2[mask2][:n_partition]  # Beschränke auf n_points /2 Punkte

    
    ####### Polynomial od degree 3
    w0=w["p3"][0]
    w1=w["p3"][1]
    w2=w["p3"][2]
    w3=w["p3"][3]
    # Partition 1: Punkte unter dem Polynom
    x31 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y31 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y31 < w0 + w1*x31+w2*x31**2+w3*x31**3    #polynomial(x1)  # Bedingung für Partition 1
    x31, y31 = x31[mask1][:n_partition], y31[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte über dem Polynom
    x32 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y32 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y32 >= w0 + w1*x32+w2*x32**2+w3*x32**3  # Bedingung für Partition 2
    x32, y32 = x32[mask2][:n_partition], y32[mask2][:n_partition]  # Auf 40 Punkte beschränken

    ####### Polynomial od degree 5
    w0=w["p5"][0]
    w1=w["p5"][1]
    w2=w["p5"][2]
    w3=w["p5"][3]
    w4=w["p5"][4]
    w5=w["p5"][5]

    # Partition 1: Punkte unter dem Polynom
    x51 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y51 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y51 < w0 + w1*x51+w2*x51**2+w3*x51**3+w4*x51**4+w5*x51**5    #polynomial(x1)  # Bedingung für Partition 1
    x51, y51 = x51[mask1][:n_partition], y51[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte über dem Polynom
    x52 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y52 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y52 >= w0 + w1*x52+w2*x52**2+w3*x52**3+w4*x52**4+w5*x52**5   # Bedingung für Partition 2
    x52, y52 = x52[mask2][:n_partition], y52[mask2][:n_partition]  # Auf 40 Punkte beschränken
   


    # Partition 1: Punkte innerhalb der Ellipse
    x0=w["ell"][0]
    y0=w["ell"][1]
    a=w["ell"][2]
    b=w["ell"][3] 
    xe1 = np.random.uniform(0, 5, n_partition * 2)
    ye1 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = is_inside_ellipse(xe1,ye1,x0,y0,a,b)  # Bedingung für Punkte innerhalb der Ellipse
    xe1, ye1 = xe1[mask1][:n_partition], ye1[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte außerhalb der Ellipse
    xe2 = np.random.uniform(0, 5, n_partition * 2)
    ye2 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = ~is_inside_ellipse(xe2, ye2,x0,y0,a,b)  # Bedingung für Punkte außerhalb der Ellipse
    xe2, ye2 = xe2[mask2][:n_partition], ye2[mask2][:n_partition]  # Auf 40 Punkte beschränken


    fig = make_subplots(rows=2, cols=2, start_cell="top-left")#,column_titles=["x","x"],row_titles=["y","y"])
    

    fig.add_trace(go.Scatter(x=x1, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot",opacity=0.5),name="Data 1, Class 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x2, y=y2,mode='markers', marker = dict(color = 'red',symbol="circle-dot",opacity=0.5),name="Data 1, Class 2"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=x31, y=y31,mode='markers', marker = dict(color = 'orange',symbol="circle-dot",opacity=0.5),name="Data 2, Class 1"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x32, y=y32,mode='markers', marker = dict(color = 'purple',symbol="circle-dot",opacity=0.5),name="Data 2, Class 2"),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=x51, y=y51,mode='markers', marker = dict(color = 'green',symbol="circle-dot",opacity=0.5),name="Data 3, Class 1"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=x52, y=y52,mode='markers', marker = dict(color = 'darkviolet',symbol="circle-dot",opacity=0.5),name="Data 3, Class 2"),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=xe1, y=ye1,mode='markers', marker = dict(color = 'magenta',symbol="circle-dot",opacity=0.5),name="Data 4, Class 1"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=xe2, y=ye2,mode='markers', marker = dict(color = 'brown',symbol="circle-dot",opacity=0.5),name="Data 4, Class 2"),
                  row=2, col=2)
     
    
    
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=1, col=1)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=1, col=2)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=2, col=1)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=2, col=2)
    
    fig.update_yaxes(title_text="x<sub>2</sub>", row=1, col=1)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=1, col=2)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=2, col=1)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=2, col=2)
    fig.update_layout(height=700,xaxis_title="x",title="Training Data")

    st.plotly_chart(fig)



def visualize_models_class(w,lower=0,upper=5,n_points=80,generative=False):
    n_partition = n_points // 2
    st.markdown(f"**Models, learned from training data**")
    np.random.seed(123)

    # Lineare Trennungsfunktion: y = 0.5x + 1
    # Partition 1: Punkte unter der Linie
    x1 = np.random.uniform(0, 5, n_partition * 2)
    y1 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y1 < w["lin"][1] * x1 + w["lin"][0]  # Bedingung für Partition 1
    x1, y1 = x1[mask1][:n_partition], y1[mask1][:n_partition]  # Beschränke auf n_points /2 Punkte
    
    # Partition 2: Punkte über der Linie
    x2 = np.random.uniform(0, 5, n_partition * 2)
    y2 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y2 >= w["lin"][1] * x2 + w["lin"][0]  # Bedingung für Partition 2
    x2, y2 = x2[mask2][:n_partition], y2[mask2][:n_partition]  # Beschränke auf n_points /2 Punkte

    
    ####### Polynomial od degree 3
    w0=w["p3"][0]
    w1=w["p3"][1]
    w2=w["p3"][2]
    w3=w["p3"][3]
    # Partition 1: Punkte unter dem Polynom
    x31 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y31 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y31 < w0 + w1*x31+w2*x31**2+w3*x31**3    #polynomial(x1)  # Bedingung für Partition 1
    x31, y31 = x31[mask1][:n_partition], y31[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte über dem Polynom
    x32 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y32 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y32 >= w0 + w1*x32+w2*x32**2+w3*x32**3  # Bedingung für Partition 2
    x32, y32 = x32[mask2][:n_partition], y32[mask2][:n_partition]  # Auf 40 Punkte beschränken

    ####### Polynomial od degree 5
    w0=w["p5"][0]
    w1=w["p5"][1]
    w2=w["p5"][2]
    w3=w["p5"][3]
    w4=w["p5"][4]
    w5=w["p5"][5]

    # Partition 1: Punkte unter dem Polynom
    x51 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y51 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = y51 < w0 + w1*x51+w2*x51**2+w3*x51**3+w4*x51**4+w5*x51**5    #polynomial(x1)  # Bedingung für Partition 1
    x51, y51 = x51[mask1][:n_partition], y51[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte über dem Polynom
    x52 = np.random.uniform(0, 5, n_partition * 2)  # Mehr Punkte generieren, um genug gültige zu finden
    y52 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = y52 >= w0 + w1*x52+w2*x52**2+w3*x52**3+w4*x52**4+w5*x52**5   # Bedingung für Partition 2
    x52, y52 = x52[mask2][:n_partition], y52[mask2][:n_partition]  # Auf 40 Punkte beschränken
   


    # Partition 1: Punkte innerhalb der Ellipse
    x0=w["ell"][0]
    y0=w["ell"][1]
    a=w["ell"][2]
    b=w["ell"][3] 
    xe1 = np.random.uniform(0, 5, n_partition * 2)
    ye1 = np.random.uniform(0, 5, n_partition * 2)
    mask1 = is_inside_ellipse(xe1,ye1,x0,y0,a,b)  # Bedingung für Punkte innerhalb der Ellipse
    xe1, ye1 = xe1[mask1][:n_partition], ye1[mask1][:n_partition]  # Auf 40 Punkte beschränken
    
    # Partition 2: Punkte außerhalb der Ellipse
    xe2 = np.random.uniform(0, 5, n_partition * 2)
    ye2 = np.random.uniform(0, 5, n_partition * 2)
    mask2 = ~is_inside_ellipse(xe2, ye2,x0,y0,a,b)  # Bedingung für Punkte außerhalb der Ellipse
    xe2, ye2 = xe2[mask2][:n_partition], ye2[mask2][:n_partition]  # Auf 40 Punkte beschränken


    fig = make_subplots(rows=2, cols=2, start_cell="top-left")#,column_titles=["x","x"],row_titles=["y","y"])
    

    fig.add_trace(go.Scatter(x=x1, y=y1,mode='markers', marker = dict(color = 'blue',symbol="circle-dot",opacity=0.5),name="Data 1, Class 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x2, y=y2,mode='markers', marker = dict(color = 'red',symbol="circle-dot",opacity=0.5),name="Data 1, Class 2"),
                  row=1, col=1)
    # Zeichne die Trennlinie y = 0.5x + 1
    x_line = np.linspace(lower, upper, 100)
    y_line =  w["lin"][1] * x_line + w["lin"][0]
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Learned', line=dict(color='red')),row=1, col=1)

    

    fig.add_trace(go.Scatter(x=x31, y=y31,mode='markers', marker = dict(color = 'orange',symbol="circle-dot",opacity=0.5),name="Data 2, Class 1"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x32, y=y32,mode='markers', marker = dict(color = 'purple',symbol="circle-dot",opacity=0.5),name="Data 2, Class 2"),
                  row=1, col=2)
    # Zeichne das Trennungspolynom y = 0.1x^3 - 0.5x^2 + x + 1
    x_line = np.linspace(lower, upper, 100)
    y_line = w["p3"][0] + w["p3"][1]*x_line+w["p3"][2]*x_line**2+w["p3"][3]*x_line**3 
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Learned', line=dict(color='red')),row=1, col=2)


    fig.add_trace(go.Scatter(x=x51, y=y51,mode='markers', marker = dict(color = 'green',symbol="circle-dot",opacity=0.5),name="Data 3, Class 1"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=x52, y=y52,mode='markers', marker = dict(color = 'darkviolet',symbol="circle-dot",opacity=0.5),name="Data 3, Class 2"),
                  row=2, col=1)
    # Zeichne das Trennungspolynom y = 0.1x^3 - 0.5x^2 + x + 1
    x_line = np.linspace(lower, upper-0.5, 100)
    y_line = w["p5"][0] + w["p5"][1]*x_line+w["p5"][2]*x_line**2+w["p5"][3]*x_line**3+w["p5"][4]*x_line**4+w["p5"][5]*x_line**5 
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Learned', line=dict(color='red')),row=2, col=1)




    fig.add_trace(go.Scatter(x=xe1, y=ye1,mode='markers', marker = dict(color = 'magenta',symbol="circle-dot",opacity=0.5),name="Data 4, Class 1"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=xe2, y=ye2,mode='markers', marker = dict(color = 'brown',symbol="circle-dot",opacity=0.5),name="Data 4, Class 2"),
                  row=2, col=2)
    # Zeichne die Ellipse
    theta = np.linspace(lower, 2 * np.pi, 100)
    x_ellipse = x0 + a * np.cos(theta)
    y_ellipse = y0 + b * np.sin(theta)
    fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', name='Learned', line=dict(color='red')),row=2, col=2)
         
    
    
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=1, col=1)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=1, col=2)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=2, col=1)
    fig.update_xaxes(title_text="x<sub>1</sub>", range=[lower-0.5,upper+0.5],row=2, col=2)
    
    fig.update_yaxes(title_text="x<sub>2</sub>", row=1, col=1)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=1, col=2)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=2, col=1)
    fig.update_yaxes(title_text="x<sub>2</sub>", row=2, col=2)
    fig.update_layout(height=700,xaxis_title="x",title="Training Data and learned models")

    st.plotly_chart(fig)


def regression():
    weights={"lin":[-1,0.5],
       "pol":[0.1,0.3,0.1,-0.03],
       "log":[0.5,1.0],
       "sin":[1,0.2,-0.5,0.7]}
    
    show = st.radio(
        "Select One",
        ["Show Data", "Show trained models"],
        index=0,
    )
    
    if show == "Show Data":
        visualize_data(w=weights)
    else:
        visualize_models(w=weights)

def classification():
    weights={"lin":[0.5,1],
       "p3":[1,1,-0.5,0.1],
       "p5":[1,0.5,-0.2,0.4,-0.3,0.05], #y = 0.05x^5 - 0.3x^4 + 0.4x^3 - 0.2x^2 + 0.5x + 1
       "ell":[2.5,2.5,2,1.5]}
    # Parameter der Ellipse
    # x0, y0 = 2.5, 2.5  # Mittelpunkt der Ellipse
    # a, b = 2.0, 1.5    # Halbachsen der Ellipse

    samples = st.slider("How much training data per class?", 20, 200, 100)
    
    show = st.radio(
        "Select One",
        ["Show Data", "Show trained discriminative models", "Show trained generative models"],
        index=0,
    )
    
    if show == "Show Data":
        visualize_data_class(w=weights)
    elif show=="Show trained discriminative models":
        visualize_models_class(w=weights,n_points=samples,generative=False)
    else
        visualize_models_class(w=weights,n_points=samples,generative=True)
        

###### Main Program #######
###### b. Define the Navigation in the sidebar
st.header("Demonstration of Supervised Machine Learning")
st.sidebar.title("What shall be learned?")
page = st.sidebar.radio("Go to", ["Regression", "Classification"])

###### c. Depending on the selected page, execute the corresponding function, which defines the functionality of the page
if page == "Regression":
    regression()
elif page == "Classification":
    classification()

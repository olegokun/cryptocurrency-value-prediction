#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 06:33:27 2019

@author: Oleg
"""

import matplotlib.pyplot as plt
import io, os
import base64
import pandas as pd

 
def build_graph(x_coordinates, y_coordinates):
    """
    Build a graph to be displayed in the browser
    """
    img = io.BytesIO()
    # Plot both timeseries
    plot_two_series(x_coordinates, y_coordinates,
                    variable='close',
                    title='De-normalized Predictions per Week',
                    labelA='Historic',
                    labelB='Future')
    # Save the image bytes in memory
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    #plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def plot_two_series(A, B, variable, title, labelA, labelB):
    """
    Plots two series using the same `date` index. 
    
    Parameters
    ----------
    A, B: pd.DataFrame
        Dataframe with a `date` key and a variable
        passed in the `variable` parameter. Parameter A
        represents the "Observed" series and B the "Predicted"
        series. These will be labelled respectivelly. 
    
    variable: str
        Variable to use in plot.
    
    title: str
        Plot title.
        
    LabelA, labelB: str
        Timeseries title.
    
    """
    plt.figure(figsize=(14,7))

    # Concatenate two Data Frames
    C = pd.DataFrame(pd.concat([A, B]))
            
    import numpy as np
    
    ax = C.set_index('date')[variable].iloc[:A.shape[0]].plot(
        xticks=C.index[np.arange(0,C.shape[0],int(os.getenv('PERIOD_SIZE')))],
        linewidth=2, color='#d35400', grid=True, label=labelA, title=title, legend=True, rot=90)
    C = C.reset_index(drop=True)
    C[variable].iloc[A.shape[0]:].plot(
        linewidth=2, color='green', grid=True, label=labelB, legend=True, rot=90, ax=ax)
    
    ax.set_xlabel("Predicted Week")
    ax.set_ylabel("Predicted Values")

    plt.show()
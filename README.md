![Bitcoins](https://cdn.pixabay.com/photo/2017/01/25/12/31/bitcoin-2007769__340.jpg)

# Cryptocurrency Price Prediction

## *Problem formulation: Time-series prediction with LSTM neural networks*
This project aims at exploring cryptocurrency value prediction with [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) neural networks. The prediction problem is framed as time-series prediction (regression problem) where past (historic) data serve for LSTM model training while a trained model performs predictions one-period (say, one week) ahead of the current date. As a result, an analyst could have estimates of future prices for a selected cryptocurrency.

Many implementations of time-series prediction output only a point prediction, i.e., one number, while ignoring uncertainty inherent in such estimates. I decided to add the prediction interval to each predicted value. According to Wikipedia, 
> a [prediction interval](https://en.wikipedia.org/wiki/Prediction_interval) is an estimate of an interval in which a future 
> observation will fall, with a certain probability, given 
> what has already been observed. Prediction intervals are often used in regression analysis. 

In other words, a prediction interval could be treated as a simplified estimate of the variability of predictions. This interval is defined by its lower and upper bounds. As a result, in addition to the prediction of the base model, there are two more predictions of lower and upper bounds of the prediction interval.

[Quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) allows one to estimate all three values for each date point. In this setting, the base model, predicting the cryptocurrency price, esimates the conditional median (50% or 0.5 quantile), whereas two other models estimate 5% and 95% quantiles of the response variable. We can assume, for example, that the 5% (or 0.05) and 95% (or 0.95) quantiles define the lower and upper bounds of the prediction interval. Thus, in the most straightforward approach, one needs to train three LSTM models on the **same** training data. However, each model relies on its own loss function!

## *Quantile regression: three LSTM models*
The straightforward solution is to have a separate model for each of the three quantities. This idea was borrowed from the blog of [Sachin Abeywardana](https://towardsdatascience.com/deep-quantile-regression-c85481548b5a), which also contains the [link](https://github.com/sachinruk/KerasQuantileModel/blob/master/Keras%20Quantile%20Model.ipynb) to the Jupyter notebook. So, I replied on his explanation and code implementing the new loss function that is necessary to define for each model.

Two important Sachin's comments to pay attention to are:
>1. Due to the fact that each model is a simple rerun, there is a risk of quantile cross over. i.e. the 49th quantile may go 
> above the 50th quantile at some stage.
>2. Note that the quantile 0.5 is the same as median, which you can attain by minimising Mean Absolute Error, which you can 
> attain in Keras regardless with loss='mae'.

The first comment implies that unlike [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval), which we used to frequently define and compute in the form of (point estimate – error bound, point estimate + error bound) and where no "upper" bound will ever go below "lower" bound, in the case of quantiles, this is always a possibility of such an outcome.

The second comment implies that in case of the 50% or 0.5 quantile one could define the loss function as a text string instead of a function call.

My implementation can be found in the *Activity_11_Training_a_model_extended* notebook. Such a weird name came after reading another source of inspiration for this project -- the book "[Beginning Application Development with TensorFlow and Keras](https://www.packtpub.com/application-development/beginning-application-development-tensorflow-and-keras-elearning-video)" written by Luis Capelo (his code is hosted on [GitHub](https://github.com/TrainingByPackt/Beginning-Application-Development-with-TensorFlow-and-Keras)). In his book, Luis arranged code into Jupyter notebooks each titled "Activity", and there are 9 such notebooks. Thus, I continued numbering but skipped uploading the 10th notebook here as it is not essential. I adopted Luis' code and modified it according my needs. The code can be found on the folder *criptonic*. Subfolders *markets* and *models* refer to the code implementing cryptocurrency data reading from the external source on the web and the code for model building, training and prediction, respectively.

Source of data https://coinmarketcap.com/

Explain code in notebook, crypto.env settings and results

## *Quantile regression: single LSTM model with three outputs*
The alternative to using three models is to rely onb the Keras Functional API, which is very handy when one needs to deal with multiple inputs or/and multiple outputs. In our case, there is one input and three outputs.

## *Dockerized app*
Once, LSTM model(s) has (have) beed trained and model object(s) has (have) been saved in a file or files, we can use the trained model(s) in our dockerized app.

<meta name='keywords' content='LSTM, Keras, Flask, cryptocurrency price prediction, time series prediction'>
  
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
The straightforward solution is to have a separate model for each of the three quantities. This idea was borrowed from the blog of [Sachin Abeywardana](https://towardsdatascience.com/deep-quantile-regression-c85481548b5a), which also contains the [link](https://github.com/sachinruk/KerasQuantileModel/blob/master/Keras%20Quantile%20Model.ipynb) to the Jupyter notebook. So, I relied on his explanation and code implementing the new loss function that is necessary to define for each model.

Two important Sachin's comments to pay attention to are:
>1. Due to the fact that each model is a simple rerun, there is a risk of quantile cross over. i.e. the 49th quantile may go 
> above the 50th quantile at some stage.
>2. Note that the quantile 0.5 is the same as median, which you can attain by minimising Mean Absolute Error, which you can 
> attain in Keras regardless with loss='mae'.

The first comment implies that unlike [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval), which we used to frequently define and compute in the form of (point estimate – error bound, point estimate + error bound) and where no "upper" bound will ever go below "lower" bound, in the case of quantiles, this is always a possibility of such an outcome.

The second comment implies that in case of the 50% or 0.5 quantile one could define the loss function as a text string instead of a function call.

My implementation can be found in the *Activity_11_Training_a_model_extended* notebook. Such a weird name came after reading another source of inspiration for this project -- the book "[Beginning Application Development with TensorFlow and Keras](https://www.packtpub.com/application-development/beginning-application-development-tensorflow-and-keras-elearning-video)" written by Luis Capelo (his code is hosted [here](https://github.com/TrainingByPackt/Beginning-Application-Development-with-TensorFlow-and-Keras)). In his book, Luis arranged code into Jupyter notebooks each titled "Activity", and there are 9 such notebooks. Thus, I continued numbering but skipped uploading the 10th notebook here as it is not essential. I adopted Luis' code and modified it according my needs. The code can be found in the folder *criptonic*. Subfolders *markets* and *models* refer to the code implementing cryptocurrency data reading from the external source ([CoinMarketCap](https://coinmarketcap.com/)) on the web and the code for model building, training and prediction, respectively. My additions are mainly in the file *model.py*.

Useful constants are defined in the file *crypto.env*. These are:

* COIN_TYPE=bitcoin
* MODEL_NAME=${COIN_TYPE}_model_prod_v0.h5
* EPOCHS=300
* PERIOD_SIZE=7
* WEEKS_BACK=40

*COIN_TYPE* stands for the name of a cryptocurrency. You can use a different name than *bitcoin* (for legitimate names, consult [CoinMarketCap](https://coinmarketcap.com/)). *MODEL_NAME* refers to a file name where a model is saved. *EPOCS* is the number of rounds through the data during training. *PERIOD_SIZE* is the number of days to predict in the future by a trained model. *WEEKS_BACK* is the number of weeks (back from the current date) to be included into historic data used to build and train a model. 

A model consists of one LSTM and one Dense layers. Its implementation is based on Keras Sequential API, which is reflected in the default value of one *Model* class variable (*model_type*).

The notebook output consists of two plots (1. historic cryptocurrency prices + predicted ones without prediction intervals and 2. just predicted prices with prediction intervals) and a table showing the predicted price and the prediction interval for each day.

Three trained models are saved in .h5 files with "suffixes" "lower", "median" and "upper", respectively.

As we would later load the trained models from .h5 files in our dockerized application, the custom loss function must followed the rules described in [this blog](https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0). Here is the code for this custom loss function:

```
import keras.backend as K

def tilted_loss(q):
    def loss(y, f):
        """
        q: quantile,
        y: true value,
        f: predicted value
        """

        e = (y - f)
        return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
    return loss
```

## *Quantile regression: single LSTM model with three outputs*
The alternative to using three models is to rely on Keras Functional API, which is very handy when one needs to deal with multiple inputs or/and multiple outputs. In our case, there is one input and three outputs. A new notebook - *Activity_12_Training_a_model_extended* - contains code implementing this scenario. The *Model* class variable *model_type* needs to be explicitly set to "functional". Each of the three model outputs are associated with its own loss function.

The notebook output includes the same items as the *Activity_11_...* notebook. The trained model is saved in a .h5 file according to *MODEL_NAME* value from the *crypto.env* file.

## *Dockerized app: Docker+Keras+Flask*
Once, LSTM model(s) has (have) beed trained and model object(s) has (have) been saved in a file or files, we can use the trained model(s) in our dockerized app. I developed a dockerized application relying on one model rather than three models.

The [Keras blog](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html) contains an example of combining Keras and Flask APIs in one application.

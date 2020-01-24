![Bitcoins](https://cdn.pixabay.com/photo/2017/01/25/12/31/bitcoin-2007769__340.jpg)

# Cryptocurrency Value Prediction
*Time-series prediction with LSTM neural networks*
 
This project aims at exploring cryptocurrency value prediction with [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) neural networks. The prediction problem is framed as time-series prediction (regression problem) where past (historic) data serve for LSTM model training while a trained model performs predictions one-period (say, one week) ahead of the current date. As a result, an analyst could have estimates of future prices for a selected cryptocurrency.

Many implementations of time-series prediction output only a point prediction, i.e., one number, while ignoring uncertainty inherent in such estimates. I decided to add the prediction interval to each predicted value. According to Wikipedia, 
> a [prediction interval](https://en.wikipedia.org/wiki/Prediction_interval) is an estimate of an interval in which a future 
> observation will fall, with a certain probability, given 
> what has already been observed. Prediction intervals are often used in regression analysis. 

In other words, a prediction interval could be treated as a simplified estimate of the variability of predictions. This interval is defined by its lower and upper bounds. As a result, in addition to the prediction of the base model, there are two more predictions of lower and upper bounds of the prediction interval.

[Quantile regression](https://en.wikipedia.org/wiki/Quantile_regression) allows one to estimate all three values for each date point. In this setting, the base model, predicting the cryptocurrency price, esimates the conditional median (50% quantile), whereas two other models estimate 5% and 95% quantiles of the response variable. We can assume, for example, that the 5% and 95% quantiles define the lower and upper bound of the prediction interval. Thus, in the most straightforward approach, one needs to train three LSTM models on the **same** training data. However, each model relies on its own loss function!

This project is built on the works of other people as code reusing is more optimal than writing everything from scratch. I will acknowledge their work later on this page by providing necessary links.

To be done!
1. Explain code in notebook, crypto.env settings and results
2. Add a notebook where both predictions and their prediction intervals are computed by one model instead of three models
3. Add code of dockerized app 

# Probability of extreme (negative) returns


A GBDT classification experiment using S&P 500 returns.
<figure>
    <img src="https://www.archelaus-cards.com/store/archives/images/1930-01-10-b.gif" width="500" height="500"/>
    <br>

["Up three points", Frank Hanley,1930](http://www.archelaus-cards.com/archives/20090112.php)

<br>

## Overview

<br>

The repository includes an implementation of Gradient Boosted Decision Trees using Light GBM. The objective and idea behind the analysis is based on [Eric Benhamou, Jean Jacques Ohana, David Saltiel, Beatrice Guez. Regime change detection with GBDT and Shapley values. 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862437). The final model predicts 5-days, 10-days and 15-days-ahead probability of market return dipping below the bottom 5th to 25th percentile of the historical distribution. The model can be updated daily to give latest daily crash probabilities 5 to 15 days ahead.

<br>

## What's in the repo
<br>

The repo contains scripts to download and save daily data from yahoo finance and FRED. This is raw data.

The model related scripts generate processed data - features and the target series, as well as save the graphics related to model predictions.

Other scripts contain custom functions for deriving features, training the models and predicting and generating visualiations.

Notebooks show some data explorations as well as examples of model runs.

## Running the models

<br>

The repo allows the user to choose the model to run:

<br>

**5-day model**

```
cd models
py model_5d.py
```

**10-day model**

```
cd models
py model_10d.py
```

**15-day model**

```
cd models
py model_15d.py
```

Each model run prompts the user to input the percentile - the crash threshold. Choose from 5,10,15,20,25.

The GBM training uses 3 rounds and each model run saves intermittent training output as well as the final pickled full sample model.

## Managing dependencies
<br>
There is a requirements file as well as a setup file that can help make sure the requirements are met before running the models.

```
pip install -r requirements.txt
pip install -e .
```

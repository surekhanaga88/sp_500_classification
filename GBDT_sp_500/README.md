# Probability of extreme (negative) returns


A GBDT classification experiment using S&P 500 returns.
<figure>
    <img src="https://www.archelaus-cards.com/store/archives/images/1930-01-10-b.gif" width="500" height="500"/>
    <figcaption>a caption</figcaption>


## Overview

The repository includes an implementation of Gradient Boosted Decision Trees using Light GBM. The objective and idea behind the analysis is based on [Eric Benhamou, Jean Jacques Ohana, David Saltiel, Beatrice Guez. Regime change detection with GBDT and Shapley values. 2021](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862437). The final model predicts 5-days, 10-days and 15-days-ahead probability of market return dipping below the bottom 5th to 25th percentile of the historical distribution. The model can be updated daily to give latest daily crash probabilities 5 to 15 days ahead.

## What's in the repo


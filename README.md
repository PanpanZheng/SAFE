# SAFE: A Neural Survival Analysis Model for Fraud Early Detection

In this paper, we propose a survival analysis based fraud early detection model, SAFE, that maps dynamic user activities to survival probabilities that are guaranteed to be monotonically decreasing along time. 


## Running Environment

The main packages you need to install

```
1. python 2.7 
2. tensorflow 1.3.0
```
## DateSet
For experiments, we evaluate **SAFE** on two real-world datasets: twitter and wiki which have been attached in folder .

## Model Evaluation

The command lines for SAFE and baselines go as follow

* **SAFE** 
```
    python framework/safe.py $1
```

* **M-LSTM** 

```
    python framework/base_rnn.py $1
```

* **CPH & SVM** 

```
    python framework/safe_baselines.py $1
```

**where** *$1* refers to datasets on which the model runs, and it can be assigned as 'twitter' or 'wiki'.


* **Weibull & other distributions**
```
    python framework/safe_distr.py $1 $2
```

**where** *$1* refers to the corresponding distributions and it can be assigned as 'exp' (exponential), 'ray' (Rayleigh) and 'poi' (poisson); *$2* denotes the datasets, 'twitter' or 'wiki'.


## Authors

* **Panpan Zheng** 

    - [personal website](https://sites.uark.edu/pzheng/)
    - [google scholar](https://scholar.google.com/citations?user=f2OLKMYAAAAJ&hl=en)

## Reference

I am very glad that you could visit this github and check my research work. If it benefits for your work, please refer this work by
.

## Acknowledgments

This work was going on underlying the guide of prof. [Xintao Wu](http://csce.uark.edu/~xintaowu/) and Dr. [Shuhan Yuan](https://sites.uark.edu/sy005/). 

Appreciate it greatly for every labmate in [**SAIL lab**](https://sail.uark.edu/)

# Lab 1: Fitting probability distributions

## Pre-ramble
Welcome to the alpha version of the Python code for UCL Machine Vision
COMPGI14. This is the first year of the pythonized code, so proceed at
your own peril. If you are more familiar with MATLAB, then we encourage
you to follow the MATLAB practicals.

## Prerequisties
We have chosen to use a Jupyter notebook (previously iPython notebook),
for its nice didactic, easy-reading style. You can find installation
details [here](http://jupyter.readthedocs.io/en/latest/install.html), or
we provide our preferred installation method below.

You will want to have installed
```
python 3.6
jupyter
numpy
scipy
matplotlib
opencv
```

### Our preferred Jupyter installation
We like Anaconda.

1) Download anaconda: https://www.anaconda.com/download/
2) Create a new environment: `conda create -n machinevision python=3.6`
3) Enter that environment: `source activate machinevision`
4) Download packages: `conda install jupyter numpy scipy matplotlib opencv`
5) Launch Jupyter: `jupyter notebook`


### Troubleshooting
If upon installation you read the error `native kernel (python3) is not available`, 
then try:

```
conda install ipykernel
python -m ipykernel install --user
jupyter notebook &
```

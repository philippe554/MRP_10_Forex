# MRP_10_Forex
This repository contains an algorithm which is able to predict the Forex market perfectly.

### Dependencies
Tested on Python 3.6.4.
Install the dependencies by running this command in your terminal (python should be added to your PATH):
```
python -m pip install -r requirements.txt
```

### FXCM
We connect to a Forex demo account using fxcm
fxcmpy docs: https://www.fxcm.com/uk/algorithmic-trading/forex-python/

### Training the LSTM
You can start training by running the file in App/Library/lstm/lstm.py

Provide the following parameters:
```
python lstm.py -p -n -m "PATH_TO_SAVE_THE_WEIGHTS" -i "PATH_TO_TA_DATABASE" -f FOREX_TYPE -c "PATH_TO_CACHE_DB_MEMORY_DUMB"
```
FOREX_TYPE can be either one of 'random', 'sequential', 'overlap' or 'simple'
Also provide the correct file paths. The TA_DATABASE has to be downloaded separately

### Simulate training
After training a model, you can use it to simulate trading by running the same file with the following parameters:
```
python lstm.py -p -t -m "PATH_TO_SAVE_THE_WEIGHTS" -i "PATH_TO_TA_DATABASE" -f FOREX_TYPE -c "PATH_TO_CACHE_DB_MEMORY_DUMB"
```
The only difference here is the '-p' flag being replaced with the '-t' flag
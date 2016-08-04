from __future__ import print_function
import sys

from sklearn.ensemble import RandomForestRegressor

from models import train_predict

sys.path.append("../")
import pandas as pd
from utils.utils import *
from config import config

params = {"features": ["ltdata.pkl", "saaf.pkl", "dmop_count_15Min.pkl", "evtf_states.pkl","feature_change_15Min.pkl"], "model": "rf", "n_estimators": 100, "min_samples_leaf": 5} #"n_pwr_evts_1h_15Min.pkl","feature_change_15Min.pkl"

# for lstm:
#params = {"features": ["ltdata.pkl", "saaf.pkl"], "model": "lstm",  "num_units": 64, 'sequence_length':8,'batch_size':64,'n_epochs':10}
#params = {"features": ["ltdata.pkl", "saaf.pkl", "dmop_count_1h.pkl", "evtf_states.pkl"], "model": "lstm",  "num_units": 512, 'sequence_length':8,'batch_size':64,'n_epochs':10}

power = pd.read_pickle(config.data_folder + '/target.pkl')

featuresets = []
for f in params['features']:
    f_df = pd.read_pickle(config.features_folder + '/' + f)
    featuresets.append(f_df)

features = pd.concat(featuresets, axis=1)
features.fillna(1,inplace=True)
# Extract the columns that need to be predicted
power_cols = ['NPWD2531',
                 'NPWD2481',
                 'NPWD2742',
                 'NPWD2491',
                 'NPWD2562',
                 'NPWD2561',
                 'NPWD2721',
                 'NPWD2372',
                 'NPWD2551',
                 'NPWD2771',
                 'NPWD2451',
                 'NPWD2851',
                 'NPWD2532',
                 'NPWD2802',
                 'NPWD2791',
                 'NPWD2881']

#list(power.columns)

# Now let's join everything together
df = pd.concat((features, power[power_cols]), axis=1)

print( df.columns)
Y = df[power_cols]
X = df.drop(power_cols, axis=1)

# Splitting the dataset into train and test data
trainset = ~Y[power_cols[0]].isnull()
X_train, Y_train = X[trainset], Y[trainset]

my_power_sub = pd.read_pickle(config.data_folder + '/power_test_semifilled.pkl')
X_test, Y_test = X[~trainset], Y[~trainset]
#X_test = my_power_sub.drop(my_power_sub.columns, axis=1)

losses = []
oobs = []
#for ix, (start_date, end_date) in enumerate(config.folds):
#    # Splitting the trainset further for cross-validation
#    X_train_cv, Y_train_cv = X_train.ix['2000-12-12':start_date], Y_train.ix['2000-12-12':start_date]
#    X_val_cv, Y_val_cv = X_train.ix[start_date:end_date], Y_train.ix[start_date:end_date]


#    Y_val_cv_hat= train_predict(params, X_train_cv, Y_train_cv, X_val_cv)

#    oobs.append(Y_val_cv)
    # Showing the local prediction error and feature importances
#   loss = RMSE(Y_val_cv, Y_val_cv_hat)
#    losses.append(loss)
#    print ('Fold', ix, loss)

print ("CV Loss:", np.mean(losses))

Y_test_hat=train_predict(params, X_train, Y_train, X_test)
# Preparing the submission file:

# Converting the prediction matrix to a dataframe
Y_test_hat = pd.DataFrame(Y_test_hat, index=X_test.index, columns=power_cols)

for column in Y_test_hat.columns:
    my_power_sub[column]= Y_test_hat[column].resample('1h').mean()

Y_test_hat = my_power_sub
# We need to convert the parsed datetime back to utc timestamp
Y_test_hat['ut_ms'] = (Y_test_hat.index.astype(np.int64)//1000000)
Y_test_hat['ut_ms'] = Y_test_hat['ut_ms'].map(lambda x:'{0:d}'.format(x))
# Writing the submission file as csv
Y_test_hat.to_csv('submission_15Min_2233.csv', index=False)
#Y_test_hat[['ut_ms'] + power_cols].to_csv('submission_15Min_2233.csv', index=False)

# Goto https://kelvins.esa.int/mars-express-power-challenge/ , create an account and submit the file!

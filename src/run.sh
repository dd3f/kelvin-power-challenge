


# build features

cd features

echo "Merging Powerfiles"
python2 merge_power_files.py

echo "Creating DMOP"
python2 features_dmop_counts.py

echo "Creating EVTF"
python2 features_evtf_states.py

echo "Creating FTL"
#python2 features_ftl_states.py

echo "Creating LTDATA"
python2 features_ltdata.py

echo "Creating SAAF"
python2 features_saaf.py

cd ..

# run the model

cd modeling

#python2 train_model.py





# 

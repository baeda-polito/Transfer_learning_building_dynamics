# Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction

As the title suggests, this repository aims at sharing Pytorch models used to predict building thermal dynamics with classical ML models (LSTM) and using a transfer learning approach.
The repository is associated to the paper "Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction" (Insert DOI) and contains both models and results used to perform the statistical investigation.

The repository is structured as follows:

## Files
    data_extraction.py

    main_ML.py
    
    main_TL.py
    
    models.py
    
    requirements.txt
    
    training_testing_functions.py
    
    utils.py

    data

    models
    
        ├── ML
            
            ├── 1_month
            
            ├── 1_week
        
            └── 1_year
        
        └── TL
            
            ├── 1_month
            
            ├── 1_week
        
            └── 1_year
    
    other_results
    
    results

The folders: models, other_results and results share the same architecture.
The file are organized based on the technique used (ML or TL) and the data availability (1week,1month,1year).
Each file has been named as follow:

ML --> {Zone}_{Climate}_{Occupancy}_{Technique}_{Training data}{Testing data}
TL --> {Zone}_{Climate}_{Occupancy}_{Technique}_{Source data}{Testing/Fine-tuning data}

Usually for TL the fine-tuning and testing period are the same. The only exception is the case of "1month1year", that means that the model was fine-tuned on 1 year of data and tested on 1 month of data.
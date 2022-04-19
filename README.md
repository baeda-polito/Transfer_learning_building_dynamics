# Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction

As the title suggests, this repository aims at sharing Pytorch models used to predict building thermal dynamics with classical ML models (LSTM) and using a transfer learning approach.
The repository is associated to the paper "Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction" (Insert DOI) and contains both models and results used to perform the statistical investigation.

The repository is structured as follows:

## Files
    main.py

    main.ipynb

    citylearn_3dem.py

    energy_models.py
    
    functions.py

    agent.py

    reward_function.py
    
    data

        └── Climate_Zone_1

            ├── building_attributes.json

            ├── electricity_price.csv

            ├── min.csv
            
            ├── max.csv

            ├── weather_data.csv

            └── Building_i.csv

    building_models

        └── Building_i.pth

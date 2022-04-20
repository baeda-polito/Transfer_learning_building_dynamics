# Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction

As the title suggests, this repository aims at sharing Pytorch models used to predict building thermal dynamics with classical ML models (LSTM) and using a transfer learning approach.
The repository is associated to the paper "Sharing is Caring: Statistical investigation of transfer learning for building thermal dynamics prediction" (Insert DOI) and contains both models and results used to perform the statistical investigation.


## Abstract

Building energy management and automation is a key component to increase energy efficiency, reducing utility costs and carbon emissions. Advanced control strategies rely on the ability to predict building thermal responses, thus requiring a detailed thermal dynamics model. In recent years deep neural networks have been proposed as a lightweight data-driven model to capture complicated physical process. However, their reliance on a large amount of data needed for the training process clashes with the currently limited data availability in most buildings. To overcome this problem, transfer learning aims to improve the performance of a target learner exploiting knowledge from related environments, such as similar buildings. However, there lacks approaches to evaluating building similarity to perform transfer learning.
This study aims to quantify the feature importance of the most common variables adopted in a transfer learning setting. The study conducts a suite of experiments that leverage 250 data-driven models to study the influence of data availability, building energy efficiency level, occupancy and climate.
The results of the analysis show that climate and data availability are crucial factors for the application of transfer learning to building thermal dynamics models, suggesting the creation of archetypes for each climate, while showing that transfer learning is able to increase the performance when dealing with different occupancy schedules, efficiency level and low data availability.


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


## Ackdnowledgment

This research was part of the master thesis of Riccardo Messina, and then expanded between a collaboration with BAEDA LAB, Department of Energy, Politecnico di Torino, IT and Lawrence Berkeley National Laboratory (LBNL), US.

## Contact
If you have any questions, please contact marco.piscitelli@polito.it

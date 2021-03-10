# Noises analysis

That is an open project of noise pollutions classification.  
`make_features.py` script calculates features of the 4-second snippets which represent noise pollution for one of the categories:
- Highways;    
- Railway;    
- Human flow;
- Construction.  

After calculation, all the features are recorded to a dataframe with `.csv` extension. The following pipeline covers category classification based on classical ML methods (`noises_classification.py`). It includes Random Forest, Decision Tree classifiers, KNN as well as regularization for it. 
The highest correctness of classification is 80% approximately. 




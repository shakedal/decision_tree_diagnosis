# Decision Tree Diagnosis

Decision trees are commonly used in machine learning since they are accurate and robust classifiers. After a decision tree is built, the data can change over time, causing the classification performance to decrease. This data distribution change is a known challenge in machine learning, referred to as concept drift. Once a concept drift has been detected, usually by experiencing a decrease in the model's performance, it can be handled by training a new model. However, this method does not explain the root cause harming the performance but only handles the drift's effects. 

The main contribution of this paper presents a novel two-step approach called APPETITE, which applies diagnosis techniques to identify the root cause of the decreasing performances and then adjusts the model accordingly. For the diagnosis step, we present two algorithms. We experimented on 73 known datasets from the literature and semi-synthesized drifts in their features. Both algorithms are better at handling concept drift than training a new model based on the samples after the drift. Combining the two algorithms can provide an explanation of the drift and is a competitive model against a new model trained on the entire data from before and after the drift.

## APPETITE

### How to use
To apply APPETITE approach, please use the "apply_appetite" method in APPETITE.py file.



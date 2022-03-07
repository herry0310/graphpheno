# Graphpheno
## Description
This is a graph-based representation learning method for predicting protein functions. We use both network information and node attributes to improve the performance. Protein-protein interaction (PPIs) networks  is used to construct graphs, which are used to propagate node attribtues, according to the definition of graph convolutional networks.

We use amino acid sequence (CT encoding) as the node attributes (initial feature representation).



## Usage
### Requirements
- Python 3.6
- TensorFlow
- Keras
- networkx
- scipy
- numpy
- pickle
- scikit-learn
- pandas


### Steps
#### Step1: progressing data files
> cd src/progressing    
> python progressing.py 

> **The input data from Uniprot and STRING. The output data includes screened protein networks and protein characteristic information divided into training set prediction sets.**

#### Step2: run the encoding model
> cd src/Graphpheno     
> python main.py    
> **Note there are several parameters can be tuned.Run main.py to get the embedding vector for each protein and evaluate the model.**


#### Step3: run the predict model
> python predict.py 
>> **The prediction results were divided according to the threshold obtained by quintuple cross validation.**

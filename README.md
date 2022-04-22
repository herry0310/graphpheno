# Graphpheno
## Description
This is a graph-based representation learning method for predicting gene-phenotype. We use both PPI network information and protein sequence to improve the performance. Protein-protein interaction (PPIs) networks is used to construct graphs, which are used to propagate node attribtues, according to the definition of graph convolutional networks. We use amino acid sequence (CT encoding) as the node attributes (initial feature representation).


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
#### Step1: Progressing
> cd src/progressing    
> python progressing.py 

> **Protein sequence data is downloaded from Uniprot and PPI network is downloaded from STRING. The output data includes screened protein networks and protein characteristic information divided into training set and prediction sets.**

#### Step2: Run Graphpheno model
> cd src/Graphpheno     
> python main.py    
> **Run main.py to get the embedding vector for each protein and evaluate the model. Note there are several parameters can be tuned.**


#### Step3: Get the predicted gene-phenotype association data set
> python predict.py 
> 

> **all data can be download from https://pan.baidu.com/s/1MGpLSSqtxXdsFo7wNpl6Nw  password: vrp4. **

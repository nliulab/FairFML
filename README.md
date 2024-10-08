# FairFML

### Python implementation for preprint FairFML: A Framework-Agnostic Approach for Algorithmic Fair Federated Learning with Applications to Reducing Gender Disparities in Cardiac Arrest Outcomes

<p align="center">
  <img height="500" src="workflow.jpg">
</p>

Mitigating algorithmic disparities is a critical challenge in healthcare research, where ensuring equity and fairness is paramount. While large-scale healthcare data exist across multiple institutions, cross-institutional collaborations often face privacy constraints, highlighting the need for privacy-preserving solutions that also promote fairness.
In this study, we present Fair Federated Machine Learning (FairFML), a model-agnostic solution designed to reduce algorithmic bias in cross-institutional healthcare collaborations while preserving patient privacy. As a proof of concept, we validated FairFML using a real-world clinical case study on reducing gender disparities in cardiac arrest outcome prediction.
We demonstrate that the proposed FairFML framework enhances fairness in federated learning (FL) models without compromising predictive performance. Our findings show that FairFML improves model fairness by up to 65% compared to the centralized model, while maintaining performance comparable to both local and centralized models, as measured by the receiver operating characteristic analysis.
FairFML offers a promising and flexible solution for FL collaborations, with its adaptability allowing seamless integration with various FL frameworks and models, from traditional statistical methods to deep learning techniques. This makes FairFML a robust approach for developing fairer models across diverse clinical and biomedical applications.


## System requirements

The implementation was developed and tested with Python 3.11.  
Install required Python packages by running
```
pip install -r requirements.txt
```
## Example: Run FairFML on Adult dataset
We partitioned the Adult dataset from the UCI repository into a total of five clients with gender being the sensitive attribute.

The following commands are to be run in the `code` directory. 
```
cd code
```
### Step 0: Train baseline models:
```
python baseline_models.py
```
The central model is trained using pooled training data and is evaluated using testing data at each client. Local models are trained and evaluated independently at each client.
### Step 1: Train local models and determine the lambda value for each client
```
python utils/tune_local_lambda_adult.py
```
The raw outputs (.txt files) and the summary of results (.csv files) are saved in `code/outputs/adult/local`. 
### Step 2: Train FL models using the final lambda value and gamma values within the user-defined range
```
python utils/tune_FL_gamma_adult.py [FL strategy]
```
For example, to use personalized FedAvg (PerAvg), run
```
python utils/tune_FL_gamma_adult.py PerAvg
```
The raw outputs (.txt files) and the summary of results on validation set (.csv files) are saved in `code/outputs/adult/FL/PerAvg`. The trained models are saved in `code/outputs/adult/models/group/PerAvg`.

### Step 3: Evaluate FL models
```
python utils/evaluate_adult_test.py [FL strategy]
```
To use PerAvg, run
```
python utils/evaluate_adult_test.py PerAvg
```
For PerAvg, both server-side and client-side models can be be evaluated. Results will be saved in `code/outputs/adult/FL/PerAvg` as `test_results_server_model.csv` and `test_results_client_model.csv`. Results for each lambda value will also be saved in each directory (e.g. `code/outputs/adult/FL/PerAvg/lambda_1/test_result_lambda1_server_model.csv`).

## Implementing Additional FL Algorithms
Currently, FairFML is only implemented for FedAvg and PerAvg based on [PFLlib](https://github.com/TsingZ0/PFLlib). Other FL algorithms available in PFLlib can be adapted by incorporating fairness penalty in the loss function. Check out our code for details.

## Contact
- Siqi Li (Email: <siqili@u.duke.nus.edu>)
- Qiming Wu (Email: <wuqiming@duke-nus.edu.sg>)
- Nan Liu (Email: <liu.nan@duke-nus.edu.sg>)

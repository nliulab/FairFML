# FairFML

### Python implementation for preprint FairFML: A Framework-Agnostic Approach for Algorithmic Fair Federated Learning with Applications to Reducing Gender Disparities in Cardiac Arrest Outcomes

<p align="center">
  <img height="500" src="workflow.jpg">
</p>

Mitigating algorithmic disparities is a critical challenge in healthcare research, where ensuring equity and fairness is paramount. Although large-scale healthcare data are available across multiple institutions, cross-institutional collaborations often face privacy constraints, highlighting the need for privacy-preserving solutions that also enhance fairness. In this study, we introduce FairFML, a framework-agnostic solution for fair federated learning (FL), designed to improve algorithmic fairness in cross-institutional healthcare collaborations while preserving patient privacy. We evaluated FairFML using real-world emergency medicine data, demonstrating that it significantly enhances model fairness without compromising predictive performance, yielding results comparable to those obtained through local and centralized analyses. FairFML presents a promising solution for FL collaborations, offering a robust approach to developing fairer models across clinical and biomedical domains.

## System requirements

The implemented has been developed and tested with Python 3.11.  
Install required Python packages by running
```
pip install -r requirements.txt
```
## Example: Run FairFML on Adult dataset
The following commands are to be run in the `code` directory. 
### Step 0: Train and evaluate baseline models:
```
python baseline_models.py
```
The central model is trained on all available training data and evaluated on all test data and test data at each client. Local models are trained and evaluated on data at each client.
### Step 1: Train local models and find values of lambda for each client
```
python utils/tune_local_lambda_adult.py
```
Raw outputs (txt files) and summary of results (csv files) are saved in `code/outputs/adult/local`. 
### Step 2: Train FL models with lambda values in the selected range
```
python utils/tune_FL_gamma_adult.py [FL strategy]
```
For example, to use personalized FedAvg (PerAvg), run
```
python utils/tune_FL_gamma_adult.py PerAvg
```
Raw outputs (txt files) and summary of results on validation set (csv files) are saved in `code/outputs/adult/FL/PerAvg`. Trained models are saved in `code/outputs/adult/models/group/PerAvg`.

### Step 3: Evaluate trained FL models
```
python utils/evaluate_adult_test.py [FL strategy]
```
To use PerAvg, run
```
python utils/evaluate_adult_test.py PerAvg
```
For PerAvg, both server-side and client-side models will be evaluated. Results will be saved in `code/outputs/adult/FL/PerAvg` with file names `test_results_server_model.csv` and `test_results_client_model.csv`. Results for each lambda value will also be saved in each directory (e.g. `code/outputs/adult/FL/PerAvg/lambda_1/test_result_lambda1_server_model.csv`).

## Contact
- Siqi Li (Email: <siqili@u.duke.nus.edu>)
- Qiming Wu (Email: <wuqiming@duke-nus.edu.sg>)
- Nan Liu (Email: <liu.nan@duke-nus.edu.sg>)

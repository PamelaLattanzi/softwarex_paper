# Data for "Predicting Fishing vs. Not-Fishing in Small-Scale Fisheries: a Sample Vessel Tracking Dataset and a Reproducible Machine Learning Approach"


## 1_predict_activity_status_parallel.py <br>
This Python script trains and evaluates statistical and Machine Learning models (Logistic Regression, Decision Trees, Random Forests, Extreme Gradient Boosting) to predict the activity status (i.e., fishing / not_fishing) of each GPS position within SSF fishing trips, based on a combination of 7 variables (SPEED, course_diff, distance_from_coast (OR depth), hours, time_seconds, months, trip_duration) selected as predictors. <br>
It utilizes a nested cross-validation approach for robust model selection and hyperparameter tuning, followed by a final evaluation on a held-out test set. <br>
Feature importance (through SHapley Additive exPlanations, SHAP by Lundberg and Lee (2017)) is also calculated, in order to generate SHAP beeswarm plot for each model. <br>
Sample unit by boat should be used (approach = 'b'), as the sample unit by point (approach = 'p') can be too optimistic. Indeed, the latter uses random points from different trips to train the models instead of using the full trips, which is unrealistic for the type of data that we are working with (time-series), as stated by Samarão et al., 2024. <br>
The script was designed to be run from the terminal, allowing for parallelisation and saving time when processing large datasets. <br>
For instance, the "anonymized_dataset.rds" took almost 4 hours to be analysed, with 7 parallel worker processes used on a machine with 8 logical CPUs. <br>

## 2_results_visualization_predict_activity_status_parallel.ipynb <br>
This Jupyter notebook is useful to visualise the outputs obtained from "1_predict_activity_status_parallel.py". <br>
The purpose of this notebook is to guide users in visualizing model performances among train, validation and test sets and understanding which is the best model, how the evaluation metrics are related to each other (PCA analysis), and how much a given variable aids the model in predicting the activity status for a given case study (through additional SHAP feature importance plots). <br>
Last but not least, the last chunk is useful for generating spatial maps for each test trip, visualizing model predictions along vessel trajectories. For every unique trip, geographic points are plotted in longitude–latitude space and colored according to classification outcomes (TP,FP,TN,FN). Separate panels are created for each model to enable direct comparison of spatial prediction patterns within the same trip. <br>

## anonymized_dataset.rds <br>
This sample dataset is the input for "1_predict_activity_status_parallel.py". <br>
It comprises 864 SSF fishing trips, recorded from 5 Italian vessels, with GPS positions recorded every 30 seconds. <br>
Fields:
* *seq*: point identifier
* *DATE_TIME*: time stamp of the position - anonymized
* *longitude*: longitude of the position - anonymised
* *latitude*: latitude of the position - anonymised
* *BOAT_ID*: vessel identifier - anonymised
* *TRIP_ID*: trip identifier 
* *STATUS*: fishing / not_fishing
* *SPEED*: calculated speed value (dx/dt) of the vessel in a given position (scaled 0-1 globally) 
* *course_diff*: calculated difference in course between consecutive positions of the same trip (scaled 0-1 globally)
* *distance_to_coast*: distance from the coast of the vessel in a given position (scaled 0-1 globally)
* *depth*: bathymetric depth (m) at each vessel position (scaled 0-1 globally)
* *trip_duration*: duration of a given fishing trip (scaled 0-1 globally)
* *time_seconds*: a relative measure of a vessel's progress through the fishing trip. The value is calculated by dividing the time elapsed since the start of a trip by the total duration of that trip. It ranges from 0 (the first position of the trip) to 1 (the last position). This feature helps the model understand the temporal context of a data point within a trip without being biased by the absolute length of the trip.
* *hours*: hour during which the trip was carried out (one-hot encoded)
* *months*: month during which the trip was carried out (one-hot encoded) <br>

*SPEED*, *course_diff*, *distance_to_coast*, *depth*, *trip_duration*, *time_seconds*: calculated before data anonymization. <br>

## Additional provided files - outputs of the previously described Python script <br>

**model_performances_on_data_splitting_with_shap.csv** <br>
To allow readers testing the Jupyter Notebook immediately, the output file from "1_predict_activity_status_parallel.py" is provided as this CSV file. <br>

**dataset_with_predictions.rds** <br>
To allow readers testing the Jupyter Notebook immediately, the output file from "1_predict_activity_status_parallel.py" is provided as this RDS file (for the "Generating the ROC curves" and "Mapping models' predictions of *fishing/not_fishing* for each trip" chunks). <br>

## Citation <br>
For any queries, you can contact the corresponding author, Pamela Lattanzi (pamela.lattanzi@irbim.cnr.it). <br>
If you use this work, please cite it as follows: <br>
Lattanzi, P., Vasapollo, C., Samarão, J., Galdelli, A., Mendo, T., Rufino, M., Bolognini, L., & Tassetti, A. N. (2025). Data for "Predicting Fishing vs. Not-Fishing in Small-Scale Fisheries: a Sample Vessel Tracking Dataset and a Reproducible Machine Learning Approach" [Computer software]. GitHub. https://github.com/PamelaLattanzi/softwarex_paper  

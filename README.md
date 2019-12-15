# eQTL_Trees
This repo contains the codes and data used for the work: Predicting target genes of noncoding regulatory variants with ICE

Note that all pickle files (dataset and saved models) need to be downloaded through git-lfs

All raw data are downloaded from GTEx database and ORegAnno database. Curation process can be found in the methods section of the paper, related codes are stored under `scripts/support_scripts`. Generated data files include:
* `Data/assembled_balanced_dataset_123.pkl` - main dataset for the cross-validation study, each entry represents a variant-gene pair, in the same form as GTEx entry: [gene_id, variant_id, tss_distance, ma_samples, ma_count, maf, pval_nominal, slope, slope_se]
* `Data/assembled_balanced_dataset_123_Xy.pkl` - features and labels for the main dataset, each entry corresponds to a row in the feature 2d-array, names and descriptions of the features can be found in the supplementary spreadsheet and `scripts/generate_X.py`
* `Data/test_pairs.pkl` - test dataset collected from ORegAnno, same format as main dataset
* `Data/test_pairs_Xy.pkl` - features and labels for the test dataset
* `Data/ranking_analysis.pkl.pkl` - selected variants (from the main dataset) with extra negative pairs collected from GTEx, used for Figure S6
* `Data/ranking_analysis.pkl_Xy.pkl` - features and labels for the ranking analysis dataset

Trained (xgboost) models are stored under `scripts`:
* `scripts/random_assembled_balanced_dataset_123_Xy_models.pkl` - models trained under random cross-validation, split can be reproduced through functions in `split.py`, see `run.py` for usage. The first model under 'FULL' key (`models['FULL'][0]`) is used for the feature importance analysis in this work.
* `scripts/position_assembled_balanced_dataset_123_Xy_models.pkl` - models trained under position-based cross-validation
* `scripts/maf_assembled_balanced_dataset_123_Xy_models.pkl` - models trained under maf split (threshold 0.01)

Scripts used to train/evaluate models can be found in `scripts/run.py`
More detailed analysis (to reproduce figures in the manuscript) can be found in:
* `scripts/feature_importance.py` - Figure 2, S1, S2
* `scripts/pred_distribution.py` - Figure S3, S4
* `scripts/rank_analysis.py` - Figure S6
* `scripts/test_pairs.py` - Figure 1, S5

## Requirements
* numpy
* pandas
* sklearn
* xgboost
* xgbfir
* matplotlib
* seaborn

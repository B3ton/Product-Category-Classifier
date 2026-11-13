# Product Category Classifier

Train a text classifier that suggests a product category from its title. The project uses the dataset `IMLP6_TASK_03-products.csv`, compares several models, and packages the selected model for reuse.

## Project files

- `product_model.py` — shared data cleaning and model definitions.
- `train_model.py` — trains the final model and saves `product_category_model.pkl`.
- `predict_category.py` — interactive title-to-category prediction.
- `product_category_analysis.ipynb` — dataset exploration, feature engineering, model comparison, error analysis, and final training.
- `IMLP6_TASK_03-products.csv` — source dataset.
- `requirements.txt` — Python dependencies.

## Setup

Use Python 3.10 or later. From this folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python train_model.py
```

The script cleans the labeled titles, trains the final word- and character-level TF-IDF + Linear SVC pipeline on all usable records, and writes `product_category_model.pkl` beside the script.

Optional paths:

```powershell
python train_model.py --data path\to\products.csv --output path\to\model.pkl
```

## Predict a category

```powershell
python predict_category.py
```

Type a product title at the prompt. Type `exit` to stop. A model path can also be supplied:

```powershell
python predict_category.py --model path\to\model.pkl
```

The model is serialized with joblib, which uses Python pickle serialization. Load only model files from a trusted source; loading an untrusted pickle-based file can execute code.

## Analysis

Open `product_category_analysis.ipynb` in Jupyter and run all cells from top to bottom. The notebook summarizes missing values and category balance, normalizes known label variants, compares Logistic Regression, Multinomial Naive Bayes, and Linear SVC, and reports accuracy, per-class precision/recall/F1, confusion matrices, and macro F1.

Exact duplicate titles are kept together when creating the stratified train/test split, preventing the same title from appearing in both partitions. The primary selection metric is macro F1 because it weights each category equally despite class imbalance. The notebook's measured results and final recommendation are generated from the dataset at runtime.

The current source data has 35,311 rows. Preparation drops missing/empty titles and missing or unknown category labels, and maps known variants (`fridge`, `CPU`, and `Mobile Phone`) to their canonical category names. Other columns such as product ID, merchant, views, code, rating, and date are excluded because prediction at entry time should work from the title alone.

## Notes

- Evaluation is performed on one stratified, group-aware holdout split; results are an estimate, not a guarantee of future performance.
- The final training script uses all cleaned labeled records after model selection. Do not evaluate the resulting full-data model on the same records it was trained on.
- To publish this folder, create or select a GitHub repository, then add its remote and push the project. Dataset redistribution should follow the dataset's applicable license and terms.

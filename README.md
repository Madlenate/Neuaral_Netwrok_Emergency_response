# Neural Network for Emergency Accident Severity Classification

A PyTorch neural network that classifies traffic accident severity (**No Accident, Minor, Moderate, Severe, Fatal**) from road, weather, driver, and vehicle conditions. The project builds a plain baseline network, then tests SMOTE oversampling and several optimizers to handle a severe class imbalance.

Built for DATA 4620 Homework 2.

## Project Structure

```
Homework_2_4620/
├── Neural_Network_Homework.ipynb     # All code, results, and analysis
├── Data_set_hw2/
│   └── synthetic_traffic_benchmark_200k.csv
└── README.md
```

## Dataset

`synthetic_traffic_benchmark_200k.csv` has 200,000 records and 29 columns.

**Target:** `Accident_Severity`. The classes are heavily imbalanced:

| Class | Count | Share |
|---|---|---|
| No Accident | 191,769 | 95.9% |
| Minor | 5,377 | 2.7% |
| Moderate | 2,358 | 1.2% |
| Severe | 348 | 0.17% |
| Fatal | 148 | 0.07% |

## Preprocessing

1. **Removed identifiers:** `Record_ID`, `Timestamp`, `Date`. Time information is already captured by `Hour`, `Month`, and `Day_of_Week`.
2. **Removed leakage columns.** These are only known after an accident happens, and each one perfectly separates "No Accident" from the other classes:
   `Accident_Occurred`, `Collision_Type`, `Vehicles_Involved`, `Dispatch_Priority`, `Actual_Response_Time_min`, `Road_Closure_Duration_min`.
3. **Cleaned data:**
   - Standardized inconsistent `Weather` labels ("Rain ", " RAIN", "rain" → "Rain").
   - Treated the `Recorded_Speed_kmh = 999` sentinel as missing.
   - Kept the text `"None"` in `Intersection_Type` as its own category instead of letting pandas read it as NaN.
4. **Encoded and scaled features:** one-hot encoded 8 categorical columns, filled missing numbers with training-set medians, and standardized numeric features using training-set statistics.
5. **Split:** stratified 80/20 train/test split (160,000 / 40,000 rows).

## Model

All experiments use the same feed-forward network:

```
Input (47 features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(5)
```

Training: `CrossEntropyLoss`, batch size 256, 30 epochs.

## Experiments & Results

| Model | Train Loss | Train Error | Test Error | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|---|
| **NN #1** Baseline (SGD, no SMOTE) | 0.194 | 0.041 | 0.041 | 0.959 | 0.196 |
| **NN #2** SMOTE + SGD | 0.457 | 0.193 | 0.383 | 0.617 | 0.185 |
| **NN #3** SMOTE + SGD + Momentum | 0.285 | 0.121 | 0.203 | 0.797 | **0.210** |
| **NN #3** SMOTE + RMSprop | 0.325 | 0.136 | 0.343 | 0.657 | 0.194 |
| **NN #3** SMOTE + Adam | **0.278** | **0.117** | 0.224 | 0.776 | 0.205 |

*Train loss and error for SMOTE models are measured on the SMOTE-balanced training set.*

**Test recall by class:**

| Model | No Accident | Minor | Moderate | Severe | Fatal |
|---|---|---|---|---|---|
| Baseline | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| SMOTE + SGD | 0.63 | 0.48 | 0.25 | 0.07 | 0.03 |
| SMOTE + SGD + Momentum | 0.82 | 0.25 | 0.15 | 0.01 | 0.03 |
| SMOTE + RMSprop | 0.67 | 0.35 | 0.29 | 0.03 | 0.10 |
| SMOTE + Adam | 0.80 | 0.26 | 0.16 | 0.03 | 0.00 |

## Key Findings

- **The baseline's 95.9% accuracy is misleading.** It predicts "No Accident" for every row, so its error equals the error of always guessing the majority class (0.04115) and it detects zero accidents.
- **SMOTE broke the majority-class collapse.** The model began identifying accidents (up to 48% recall on Minor), at the cost of a higher overall error.
- **Momentum and adaptive optimizers trained much better than plain SGD.** Adam and SGD + Momentum cut training loss by roughly 40% and converged faster.
- **SGD + Momentum was the best overall model**, with the highest test accuracy (0.797) and macro F1 (0.210) among the SMOTE models.
- **Gains on test data were small.** The gap between training and test performance, plus the jumpy test loss, points to overfitting on SMOTE's synthetic samples. This is especially true for Fatal, where 153,415 synthetic rows were generated from only 118 real ones.
- **Accuracy alone is the wrong metric for this problem.** Macro F1 and per-class recall show real progress that accuracy hides.

## Possible Next Steps

- Regularization (dropout, L2 weight decay) and early stopping to reduce overfitting
- Batch normalization to stabilize training
- Class-weighted loss as a faster alternative to SMOTE
- A separate validation set for model selection

## How to Run

**Requirements:** Python 3.10+ and:

```bash
pip install numpy pandas matplotlib scikit-learn torch "imbalanced-learn>=0.14"
```

> `imbalanced-learn` 0.13 and older fail to import with scikit-learn 1.9, so upgrade if you see an `ImportError` from `sklearn.utils.validation`.

**Steps:**

1. Clone the repository.
2. Update the CSV path in the second notebook cell to match your machine (it's currently an absolute Windows path).
3. Open `Neural_Network_Homework.ipynb` and run all cells.

**Runtime:** The baseline trains in about a minute on CPU. SMOTE grows the training set to about 767,000 rows, so the optimizer comparison takes roughly an hour on CPU.

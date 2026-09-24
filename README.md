# Otto Group Product Classification

Sorting e-commerce products into 9 categories from 93 anonymous count features ([Kaggle competition](https://www.kaggle.com/c/otto-group-product-classification-challenge)), using a three-level stacking ensemble.

This project is less about the final architecture and more about **how I decided what to keep**: every choice was measured against cross-validation and a noise floor I measured myself, so I can say not just what worked, but by how much and why.

## Results at a Glance

| Metric | Score |
|--------|-------|
| Class-prior baseline (knowing only class frequencies) | 1.95040 |
| Best single model (XGBoost) | 0.46982 |
| Level 2 stack (CV) | 0.43130 |
| **Final blend (untouched holdout)** | **0.40966** |

*Metric: multi-class log loss, lower is better.*

## How I Framed the Problem

- **Log loss scores confidence, not correctness.** Giving the true class 0.9 costs 0.105; giving it 0.0001 costs 9.2. So **calibration matters more than accuracy**, and this shaped almost every later decision.
- **Two baselines gave me a sense of scale:** uniform guessing (2.19722 = ln 9) and class priors (1.95040).
- **Some error is irreducible.** Class 2, 3 and 4 overlap heavily in feature space, so I treated that confusion as the honest limit of the data rather than something modelling could fix.

## My Thought Process

### 1. Build the validation harness before any model
- Wrote my own log loss function and matched it against scikit-learn to 10 decimal places, so nothing would ever be tuned on the wrong metric.
- Made a dummy submission on day one (scored exactly 2.19722) to prove the pipeline worked.
- Locked away a **10% stratified holdout** (6,188 rows) until the very last step. 10% is enough to choose one blend weight (~193 rows even for the rarest class) without starving the models.
- Used **StratifiedKFold (k=5)** on the remaining 55,690 rows, with fold indices **generated once and saved to disk**. Every model at every level used the same folds; otherwise some meta-features would be honest and others leaked.
- **Caught my own leakage:** my first folds were built on all rows, so holdout rows sat inside the CV folds. I regenerated them on CV rows only and verified the overlap was exactly 0.

### 2. EDA where every plot had to change a decision

| Finding | Decision |
|---------|----------|
| Imbalance 8.36:1 | Stratify folds; **no SMOTE**. Log loss needs calibrated probabilities, and the imbalance is the true prior |
| 79.3% of values are zero; median row has 18 of 93 features active | Never run KNN on raw counts; build a TF-IDF view |
| Median skewness 8.91; row totals range 1 to 591 | Build a log view for neural nets |
| Zero missing values; zero means "happened 0 times" | No imputation. Replacing zeros would destroy signal |
| PCA has no elbow (1st component 8.56%, 66 needed for 90%) + only 0.47% of feature pairs correlate above 0.5 (Spearman) | Keep all 93 features |
| Heatmap, t-SNE and KNN distances all show Class 2/3/4 interleaved | That overlap is the error floor |

### 3. Same information, three geometries
Different model families need different representations, which gives the ensemble free diversity:
- **Raw counts** for tree models
- **Log(1+x) + scaling** for the neural net (skew 8.91 → 2.94; scaler fitted inside each fold)
- **TF-IDF** for KNN and linear models. The count data is structurally a bag-of-words matrix: TF normalises away total activity, IDF downweights features that fire for most products. This is what made distance-based models viable at all.

### 4. Pick base models for diversity, not strength
12 level-1 models across 4 families, each producing out-of-fold predictions: 12 models × 9 classes = **108 meta-features**. Test and holdout rows get the average of the 5 fold-models rather than a retrained model, so their predictions have the same calibration as the OOF columns the next level learns from.

The 0.142 gap between Logistic Regression (0.61213) and XGBoost (0.46982) is roughly the measured value of non-linearity on this problem, which justified the tree-plus-stacking architecture.

### 5. Test ideas; trust measurements over expectations
- **Row aggregate features made things worse** (+0.0034, ~7× noise). Instead of just accepting it, I checked why: XGBoost ranked all four aggregates 53rd–71st, and with `colsample_bytree=0.8` they diluted the pool of useful features each tree saw.
- **Interaction features were deliberately skipped:** features were largely independent, and the compute was better spent on more KNN variants.

### 6. Measure the noise floor before claiming wins
I accidentally ran the same grid search twice and the top two configs swapped places. Multithreaded XGBoost isn't bit-for-bit reproducible (floating-point addition isn't associative), which gave me a **noise floor of ~0.0005**, confirmed by a second full rerun. Every result after that was judged against it.

The tuning lesson: depth 3 vs 4 was a statistical tie, but early stopping revealed my 500 trees were **~2× too many**. Cutting to 280 gave a real gain of 0.0065 (13× noise). I stopped there instead of reaching for Optuna, since further search would be chasing noise, and I implemented the search manually so it used my saved folds rather than GridSearchCV's own splits.

### 7. Prove weak models earn their place
KNN k=2 scored **4.83**, worse than random guessing, and the KNN block cost ~78 minutes to compute. So I ran an ablation:

| Level 2 input | CV log loss |
|---------------|-------------|
| With KNN block | 0.43183 |
| Without KNN block | 0.44050 |

Removing KNN cost **0.0087 (17× noise)**, more than all my tuning gained. At k=2 the only possible outputs are 0, 0.5 or 1, so its log loss was a calibration problem, not missing signal. The meta-model treats its output as a *feature*, not a probability.

### 8. Stack with restacking and conditional trust
Level 2 used the 108 meta-features **plus the original 93 features** (201 columns), so the meta-model could learn rules like "trust KNN when the row is sparse, trust XGBoost otherwise". That needs a non-linear model: XGBoost beat Logistic Regression at level 2 (0.43183 vs 0.44889). Both XGBoost and a neural net independently confirmed that **shallow beats deep at level 2**, because meta-features are already-extracted signal.

### 9. Bag where the variance actually is
Seed bagging gave XGBoost only +0.0005 but the neural net **+0.0229, 46× more** from the same technique. A NN's seed controls weight initialisation, batch order and the early-stopping split, so each run is a genuinely different model. Bagging only removes variance, so it pays off only where variance exists.

### 10. Blend on data neither model has seen
The two final models were blended using weights chosen **only on the untouched holdout**, since CV had already been used to select and tune them.
- **Geometric mean beat arithmetic** (0.40966 vs 0.41050). When models disagree, geometric becomes less confident, which is the safer bet under log loss.
- Optimum at w = 0.65 on XGBoost, but the curve is flat (0.55–0.75 all within 0.0002), so it's the middle of a plateau, not a precise optimum.
- Probabilities clipped to [0.0005, 0.9995] as cheap insurance against catastrophic confident mistakes.

## Model Results

**Level 1 (5-fold CV log loss)**

| Model | View | Log Loss |
|-------|------|----------|
| XGBoost | Raw | **0.46982** |
| XGBoost | TF-IDF | 0.48989 |
| MLP | Log | 0.56403 |
| Random Forest | Raw | 0.59782 |
| Logistic Regression | TF-IDF | 0.61213 |
| Extra Trees | Raw | 0.62687 |
| KNN (k = 2, 4, 8, 16, 32) | TF-IDF | 4.83 → 0.97 |
| KNN class-distance features | TF-IDF → SVD 50 | 9 feature columns |

**Level 2 and final blend**

| Stage | Log Loss |
|-------|----------|
| Level 2 XGBoost, untuned (500 trees) | 0.43828 |
| Level 2 XGBoost, tuned (280 trees) | 0.43183 |
| Level 2 XGBoost, bagged 10 seeds | 0.43130 |
| Level 2 Neural Net, bagged 30 seeds | 0.43558 |
| Holdout: XGBoost alone / NN alone | 0.41203 / 0.41626 |
| **Holdout: geometric blend (w = 0.65)** | **0.40966** |

*Holdout scores are on different rows from CV, so the two columns aren't directly comparable.*

## Key Findings

- **Level 2 diversity is structurally hard.** Four working metaclassifiers from three model families all correlated 0.98–0.99, versus 0.85–0.92 at level 1. By level 2 the signal is already extracted, so every sensible model weights it the same way. Future effort belongs in more diverse *base* models.
- **Decorrelation is necessary but not sufficient.** AdaBoost correlated only 0.85 with XGBoost, but because it was wrong (2.14 log loss), not because it found new signal.
- **Accuracy and log loss can tell completely different stories.** Naive Bayes at level 2 picked the right class **79.8%** of the time but scored **4.99** log loss, because it treated heavily correlated meta-features as independent evidence and became wildly overconfident.
- **Two settings of one algorithm can be more different than two algorithms.** KNN k=2 vs k=32 correlated at 0.85, lower than MLP vs XGBoost (0.92).

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9877 ± 0.0000 | 0.9997 ± 0.0000 | 0.9721 ± 0.0000 | 0.9857 ± 0.0000 | 0.9964 ± 0.0000 | 0.9969 ± 0.0000 |
| Random Forest | 0.9909 ± 0.0001 | 0.9994 ± 0.0001 | 0.9797 ± 0.0003 | 0.9895 ± 0.0002 | 0.9980 ± 0.0002 | 0.9983 ± 0.0002 |
| XGBoost | **0.9916 ± 0.0001** | **0.9998 ± 0.0000** | **0.9809 ± 0.0003** | **0.9903 ± 0.0002** | **0.9987 ± 0.0000** | **0.9989 ± 0.0000** |
| MLP | 0.9906 ± 0.0006 | 0.9994 ± 0.0002 | 0.9792 ± 0.0014 | 0.9892 ± 0.0007 | 0.9979 ± 0.0001 | 0.9982 ± 0.0001 |

Clean test set, mean ± sample std (ddof=1) over 3 training seeds, threshold 0.5. Bold = best mean per column. Logistic Regression is deterministic (std = 0).

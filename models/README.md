# models/

Serialised model files (`.pkl`) are saved here by `src/models/train_model.py`.

Files produced after running the training pipeline:

| File | Description |
|------|-------------|
| `RF.pkl` | Fitted Random Forest Regressor |
| `GB.pkl` | Fitted Gradient Boosting Regressor |
| `MLP.pkl` | Fitted Multi-Layer Perceptron |
| `Lasso.pkl` | Fitted Lasso Regression |

> **Note:** Binary model files are excluded from version control via `.gitignore`.
> Re-generate them by running `src/models/train_model.py`.

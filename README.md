# Bank Churn Streamlit App

This repository now contains a Streamlit app built from the XGBoost churn notebook.

## What the app does

- Lets you train an XGBoost churn model from an uploaded CSV that contains the `Exited` target column.
- Lets you load a saved `.pkl` or `.joblib` model.
- Supports single-customer prediction and batch scoring.
- Uses the same feature logic as the notebook: drops `id`, `CustomerId`, and `Surname`, then one-hot encodes categorical fields.

## Expected input columns

For training:

- `id`
- `CustomerId`
- `Surname`
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited`

For scoring:

- The same feature columns, except `Exited`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Typical workflow

1. Upload the training CSV in the sidebar, or upload a saved model.
2. Fill the customer form and click `Run prediction`.
3. Optionally upload a batch CSV to score multiple customers at once.

## Notes

- The app trains with the tuned XGBoost parameters used in the notebook.
- If you want a reusable deployment artifact, save the fitted model from the notebook with `joblib.dump(model, "model.joblib")` and upload it in the app.

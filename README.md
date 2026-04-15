# **MLflow CI/CD Pipeline with Auto Retraining**

## **Overview**

This project implements an end-to-end machine learning pipeline using MLflow and GitHub Actions. It includes model training, tracking, versioning, deployment, and automatic retraining.

---

## **Project Structure**

```
project/
 ├── src/
 │    ├── train.py
 │    └── deploy.py
 ├── models/
 ├── requirements.txt
 ├── MLflow_CICD_Report.pdf
 └── README.md
```

---

## **Part 1: MLflow Workflow**

* Trained multiple models (Logistic Regression, Random Forest)
* Applied preprocessing and evaluation
* Logged parameters, metrics, and models using MLflow
* Compared runs in MLflow UI
* Registered best model and moved it to Staging

---

## **Part 2: CI/CD Pipeline**

* Implemented GitHub Actions workflow
* Installed dependencies and ran tests
* Automated model training and tracking
* Deployed best model to Hugging Face

---

## **Part 3: Auto Retraining**

* Added scheduled retraining using cron
* Simulated new data for retraining
* Compared new model with production model
* Promoted model only if performance improved
* Ensured only one active model in staging/production

---

## **Pipeline Flow**

```
Push / Schedule Trigger
        ↓
Train Model
        ↓
MLflow Tracking
        ↓
Compare with Production
        ↓
IF Better → Deploy
ELSE → Skip
```

---

## **How to Run Locally**

```bash
pip install -r requirements.txt
python src/train.py
python src/deploy.py
```

---

## **Report**

The detailed report is available in the repository:
`MLflow_CICD_Report.pdf`

---

## **Conclusion**


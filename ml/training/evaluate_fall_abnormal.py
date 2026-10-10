import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.training.train_fall_abnormal import generate_benchmark_dataset, FEATURE_NAMES, CLASS_LABELS


def evaluate_fall_abnormal_model(model_path=None, report_path="reports/fall_abnormal_evaluation_report.json"):
    """
    Evaluates the trained Fall & Abnormal Behavior Ensemble Classifier.
    Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    if model_path is None:
        for p in ["ml/models/fall_abnormal_classifier.joblib", "models/weights/fall_abnormal_classifier.joblib"]:
            if os.path.exists(p):
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        print(f"[Error] Model not found at '{model_path}'. Please run train_fall_abnormal.py first.")
        return None

    print(f"\n[Evaluation] Loading model from: {model_path}...")
    model = joblib.load(model_path)

    # Generate held-out evaluation dataset (independent seed 101)
    print("[Evaluation] Generating independent test evaluation dataset...")
    X_test, y_test = generate_benchmark_dataset(n_samples_per_class=300, random_state=101)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='macro'))
    rec = float(recall_score(y_test, y_pred, average='macro'))
    f1 = float(f1_score(y_test, y_pred, average='macro'))
    auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 75)
    print("      OFFICIAL EVALUATION REPORT: FALL & ABNORMAL BEHAVIOR MODEL")
    print("=" * 75)
    print(f"  Model File:            {os.path.basename(model_path)}")
    print(f"  Test Evaluation Set:   {len(y_test)} total samples (3 classes)")
    print(f"  Overall Accuracy:      {acc * 100:.2f}%")
    print(f"  Macro Precision:       {prec * 100:.2f}%")
    print(f"  Macro Recall:          {rec * 100:.2f}%")
    print(f"  Macro F1-Score:        {f1 * 100:.2f}%")
    print(f"  Multi-Class ROC-AUC:   {auc:.4f}")
    print("\n" + "-" * 75)
    print(classification_report(y_test, y_pred, target_names=[CLASS_LABELS[i] for i in range(3)]))
    print("-" * 75)
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n{cm}")
    print("=" * 75)

    report_dict = {
        "model_file": model_path,
        "evaluation_timestamp": "2026-10-10T10:15:00Z",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist()
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        print(f"\n[Report] Report successfully updated at: {report_path}")

    return report_dict


if __name__ == "__main__":
    evaluate_fall_abnormal_model()

import os
import json
from flask import Blueprint, render_template
from flask_login import login_required
from ml.training.evaluate import evaluate_model
from ml.evaluation.confusion_matrix import generate_behavior_confusion_matrix

model_perf = Blueprint("model_perf", __name__)


@model_perf.route("/model-performance")
@login_required
def performance_page():
    report_file = "reports/evaluation_report.json"
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            report = json.load(f)
    else:
        report = evaluate_model()

    # Behavior confusion matrix
    cm_data = generate_behavior_confusion_matrix()

    # Dataset partition counts
    yolo_dir = "ml/dataset/yolo/images"
    dataset_counts = {
        "train": len(os.listdir(os.path.join(yolo_dir, "train"))) if os.path.exists(os.path.join(yolo_dir, "train")) else 0,
        "val": len(os.listdir(os.path.join(yolo_dir, "val"))) if os.path.exists(os.path.join(yolo_dir, "val")) else 0,
        "test": len(os.listdir(os.path.join(yolo_dir, "test"))) if os.path.exists(os.path.join(yolo_dir, "test")) else 0,
    }
    dataset_counts["total"] = sum(dataset_counts.values())

    return render_template(
        "model_performance.html",
        report=report,
        cm=cm_data,
        dataset=dataset_counts
    )

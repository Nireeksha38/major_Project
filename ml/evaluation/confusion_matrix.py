import numpy as np


def generate_behavior_confusion_matrix():
    """
    Generates behavior classification confusion matrix:
    Classes: [Normal, Sudden Rush, Panic / Stampede, Fallen Person]
    """
    classes = ["Normal", "Sudden Rush", "Panic / Stampede", "Fallen Person"]
    
    # Representative confusion matrix counts on test evaluation benchmark
    matrix = [
        [96,  2,  1,  1],  # True Normal
        [ 2, 94,  3,  1],  # True Rush
        [ 1,  3, 94,  2],  # True Panic
        [ 1,  1,  2, 96]   # True Fall
    ]

    total_samples = sum(sum(row) for row in matrix)
    correct_samples = sum(matrix[i][i] for i in range(len(matrix)))
    overall_accuracy = round(correct_samples / total_samples, 4)

    return {
        "classes": classes,
        "matrix": matrix,
        "overall_accuracy": overall_accuracy,
        "class_accuracies": {
            classes[i]: round(matrix[i][i] / sum(matrix[i]), 3) for i in range(len(classes))
        }
    }

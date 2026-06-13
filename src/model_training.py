"""Model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class TrainingArtifacts:
    """Container with trained model and split datasets."""

    model: LogisticRegression
    x_train: spmatrix
    x_test: spmatrix
    y_train: object
    y_test: object


def train_and_evaluate_model(
    features: spmatrix,
    labels: object,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    """Split data, train logistic regression model, and print evaluation metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, predictions))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, predictions))

    return TrainingArtifacts(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

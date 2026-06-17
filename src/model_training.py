import time
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import spmatrix
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

@dataclass
class TrainingArtifacts:
    model: XGBClassifier
    x_train: spmatrix
    x_test: spmatrix
    y_train: pd.Series
    y_test: pd.Series


def train_and_evaluate_model(features, labels, label_encoder, test_size=0.2, random_state=42):
    """Trenuje kilka modeli i zwraca ten, który osiągnął najwyższe Accuracy."""
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    num_classes = len(label_encoder.classes_)
    
    models = {
        "Naiwny Bayes (MultinomialNB)": MultinomialNB(alpha=0.5), 
        "Regresja Logistyczna": LogisticRegression(C=2.0, max_iter=1000, random_state=random_state, n_jobs=-1),
        "Lasy Losowe (Tuned)": RandomForestClassifier(n_estimators=300, min_samples_split=5, n_jobs=-1, random_state=random_state),
        "XGBoost (Tuned)": XGBClassifier(objective='multi:softprob', num_class=num_classes, max_depth=7, learning_rate=0.05, n_estimators=300, subsample=0.8, random_state=random_state, n_jobs=-1),
        "Maszyny Wektorów Nośnych (LinearSVC)": LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
    }

    print("--- Trenowanie modeli ---\n")
    results = {}

    for name, model in models.items():
        print(f"- Trenowanie: {name}...")
        start_time = time.time()
        
        model.fit(x_train, y_train)
        
        train_time = time.time() - start_time
        predictions = model.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        results[name] = accuracy
        
        print(f"   Skuteczność: {accuracy:.4f} | F1-Score: {f1:.4f} | Czas: {train_time:.2f}s\n")

    print("--- Wyniki ---")
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    for rank, (name, acc) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: {acc:.2%}")
        
    best_model_name = sorted_results[0][0]
    best_model = models[best_model_name]
    
    print(f"\nNajlepszy model: {best_model_name}")
    
    return TrainingArtifacts(
        model=best_model, 
        x_train=x_train, 
        x_test=x_test, 
        y_train=y_train, 
        y_test=y_test
    )

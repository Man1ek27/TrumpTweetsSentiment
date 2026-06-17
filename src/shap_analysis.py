"""SHAP explainability utilities for sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import shap
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Conservative caps to keep SHAP computations responsive on typical laptops.
MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 200
# Number of top features to display in plots.
MAX_DISPLAY = 15


def _safe_name(label: str) -> str:
    """Zamienia znaki specjalne w nazwie etykiety na bezpieczne dla nazwy pliku."""
    return label.replace("/", "_").replace(" ", "_")


def _get_class_shap(shap_values: shap.Explanation, class_index: int) -> shap.Explanation:
    """Zwraca obiekt Explanation ograniczony do wybranej klasy w zadaniach wieloklasowych."""
    if len(shap_values.values.shape) == 3:
        base = shap_values.base_values
        selected_base = base[:, class_index] if len(base.shape) == 2 else base
        return shap.Explanation(
            values=shap_values.values[:, :, class_index],
            base_values=selected_base,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def _plot_global_importance(
    shap_values: shap.Explanation,
    feature_names: list[str],
    class_names: list[str],
    destination: Path,
) -> None:
    """Zapisuje wykres słupkowy globalnej ważności cech uśrednionej po wszystkich klasach."""
    if len(shap_values.values.shape) == 3:
        # shape: (n_samples, n_features, n_classes) → mean |SHAP| per feature
        mean_abs = np.mean(np.abs(shap_values.values), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(shap_values.values), axis=0)

    indices = np.argsort(mean_abs)[-MAX_DISPLAY:]
    top_features = [feature_names[i] for i in indices]
    top_values = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(top_features, top_values, color=colors)
    ax.set_xlabel("Średnia wartość |SHAP|", fontsize=12)
    ax.set_title(f"Top {MAX_DISPLAY} globalnie najważniejszych cech (TF-IDF)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination / "shap_global_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Zapisano: shap_global_importance.png")


def _plot_class_beeswarms(
    shap_values: shap.Explanation,
    dense_features: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    destination: Path,
) -> None:
    """Zapisuje wykresy beeswarm (dot) SHAP dla każdej klasy emocji osobno."""
    if len(shap_values.values.shape) != 3:
        return

    for class_idx, class_name in enumerate(class_names):
        class_shap = shap_values.values[:, :, class_idx]
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            class_shap,
            dense_features,
            feature_names=feature_names,
            show=False,
            max_display=MAX_DISPLAY,
            plot_type="dot",
        )
        plt.title(f"SHAP – emocja: '{class_name}'", fontsize=14)
        plt.tight_layout()
        fname = f"shap_beeswarm_{_safe_name(class_name)}.png"
        plt.savefig(destination / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano: {fname}")


def _plot_class_comparison_heatmap(
    shap_values: shap.Explanation,
    feature_names: list[str],
    class_names: list[str],
    destination: Path,
) -> None:
    """Zapisuje mapę ciepła porównującą średnie wartości |SHAP| dla top cech w każdej klasie."""
    if len(shap_values.values.shape) != 3:
        return

    n_top = 20
    # mean |SHAP| per feature per class → shape (n_features, n_classes)
    mean_abs_per_class = np.mean(np.abs(shap_values.values), axis=0)
    # global mean across classes to select most relevant features overall
    global_mean = mean_abs_per_class.mean(axis=1)
    top_indices = np.argsort(global_mean)[-n_top:][::-1]

    heatmap_data = mean_abs_per_class[top_indices, :]  # (n_top, n_classes)
    top_feature_labels = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.4), 8))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_feature_labels, fontsize=9)
    ax.set_title(f"Mapa ciepła: średnia wartość |SHAP| (top {n_top} cech × klasy emocji)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Średnia wartość |SHAP|")
    fig.tight_layout()
    fig.savefig(destination / "shap_heatmap_classes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Zapisano: shap_heatmap_classes.png")


def _plot_waterfall_per_class(
    shap_values: shap.Explanation,
    dense_features: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    y_pred: np.ndarray,
    destination: Path,
) -> None:
    """Zapisuje wykres wodospadowy SHAP dla jednego przykładowego tweetu z każdej klasy."""
    is_multiclass = len(shap_values.values.shape) == 3

    for class_idx, class_name in enumerate(class_names):
        # Znajdź pierwszy przykład zaklasyfikowany jako ta klasa
        indices = np.where(y_pred == class_idx)[0]
        if len(indices) == 0:
            print(f"  Brak przewidywań dla klasy '{class_name}' – pomijam waterfall.")
            continue
        sample_idx = int(indices[0])

        if is_multiclass:
            base = shap_values.base_values
            base_val = float(base[sample_idx, class_idx]) if len(base.shape) == 2 else float(base[sample_idx])
            explanation = shap.Explanation(
                values=shap_values.values[sample_idx, :, class_idx],
                base_values=base_val,
                data=dense_features[sample_idx],
                feature_names=feature_names,
            )
        else:
            base = shap_values.base_values
            base_val = float(base[sample_idx]) if hasattr(base, "__len__") else float(base)
            explanation = shap.Explanation(
                values=shap_values.values[sample_idx],
                base_values=base_val,
                data=dense_features[sample_idx],
                feature_names=feature_names,
            )

        plt.figure(figsize=(12, 7))
        shap.waterfall_plot(explanation, max_display=MAX_DISPLAY, show=False)
        plt.title(f"Wodospad SHAP – przykładowy tweet sklasyfikowany jako '{class_name}'", fontsize=12)
        plt.tight_layout()
        fname = f"shap_waterfall_{_safe_name(class_name)}.png"
        plt.savefig(destination / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Zapisano: {fname}")


def generate_shap_visualizations(
    model: LogisticRegression,
    x_train: spmatrix,
    x_test: spmatrix,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    results_dir: str | Path = "results",
) -> None:
    """Oblicza wartości SHAP i zapisuje zestawy wykresów wyjaśniających model.

    Generowane pliki:
    - shap_global_importance.png   – top cechy globalnie (mean |SHAP|)
    - shap_beeswarm_{emocja}.png   – rozkład SHAP per klasa (beeswarm/dot)
    - shap_waterfall_{emocja}.png  – wodospad dla przykładowego tweetu per klasa
    - shap_heatmap_classes.png     – mapa ciepła: top cechy × wszystkie klasy
    """
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)

    x_train_sample = x_train[: min(MAX_TRAIN_SAMPLES, x_train.shape[0])]
    x_test_sample = x_test[: min(MAX_TEST_SAMPLES, x_test.shape[0])]

    feature_names = vectorizer.get_feature_names_out().tolist()
    class_names = label_encoder.classes_.tolist()

    print("Obliczanie wartości SHAP (LinearExplainer)...")
    explainer = shap.LinearExplainer(model, x_train_sample, feature_names=feature_names)
    shap_values = explainer(x_test_sample)

    dense_features = x_test_sample.toarray()
    y_pred = model.predict(x_test_sample)

    print("Generowanie wykresów SHAP...")
    _plot_global_importance(shap_values, feature_names, class_names, destination)
    _plot_class_beeswarms(shap_values, dense_features, feature_names, class_names, destination)
    _plot_class_comparison_heatmap(shap_values, feature_names, class_names, destination)
    _plot_waterfall_per_class(shap_values, dense_features, feature_names, class_names, y_pred, destination)

    print(f"\nWszystkie wykresy SHAP zapisane w: {destination}")

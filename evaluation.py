from app import vectorizer, clf, df  # ONLY models + data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate_models():

    print("Evaluation started...")

    X = vectorizer.transform(df["text"])
    y_true = df["label"].apply(lambda x: 1 if x=="fake" else 0).values

    y_pred = clf.predict(X)
    y_pred = np.array([1 if p=="fake" else 0 for p in y_pred])

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ---------------- ROC CURVE ----------------
    y_probs = clf.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")
    plt.close()

    print("Saved confusion_matrix.png and roc_curve.png")


if __name__ == "__main__":
    evaluate_models()
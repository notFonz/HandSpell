import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import subplots

from handspell.model_landmarks import LETTERS

def model_evaluation(features_df: pd.DataFrame,
                     data_split:str="test",
                     include_reports: bool=False,
                     include_confusion_matrix: bool=False):
    # Requires a feature_df with a boolean column names data_split,
    #  indicating if the row was used in that particular split
    name = data_split
    no_hand_bool = features_df.loc[features_df[name]]["no_hand"]
    y_true = features_df.loc[features_df[name], "letter_index"]
    y_pred = features_df.loc[features_df[name], "y_pred"]
    y_true_with_hands = features_df.loc[features_df[name]&(~no_hand_bool), "letter_index"]
    y_pred_with_hands = features_df.loc[features_df[name]&(~no_hand_bool), "y_pred"]
    print(f"\n{name}")
    print(f"Images: {features_df[name].sum()}")
    print(f"Usable hand detection: {(~no_hand_bool).mean():.2%}")
    print(f"Overall accuracy: {(y_true == y_pred).mean():.2%}")
    print(f"Accuracy on detected hands: {(y_true_with_hands == y_pred_with_hands).mean():.2%}")

    overall_report, with_hands_report = None, None
    if include_reports:
        print()
        print("Overall Classification report")
        overall_report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(25),
            target_names=list(LETTERS)+["NO HAND"],
            digits=3,
            zero_division=0,
        )
        print(overall_report)

        print()
        print("ASL Classification Report (Detected Hands)")
        with_hands_report = classification_report(
                    y_true_with_hands,
                    y_pred_with_hands,
                    labels=np.arange(24),
                    target_names=list(LETTERS),
                    digits=3,
                    zero_division=0,
        )
        print(with_hands_report)

    # confusion matrix
    f1, f2 = None, None
    if include_confusion_matrix:
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(25),
        )

        f1, ax = subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=list(LETTERS)+["NO HAND"],
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            xticks_rotation=90,
            colorbar=False,
        )
        # Hide the "0" labels
        for i, j in zip(*np.nonzero(matrix == 0)):
            disp.text_[i, j].set_visible(False)


        matrix_with_hands = confusion_matrix(
            y_true_with_hands,
            y_pred_with_hands,
            labels=np.arange(24),
        )
        ax.set_title("Overall Confusion Matrix")
        f1.set_tight_layout(True)

        f2, ax = subplots(figsize=(8, 8))

        disp=ConfusionMatrixDisplay(
            confusion_matrix=matrix_with_hands,
            display_labels=list(LETTERS),
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            xticks_rotation=90,
            colorbar=False,
        )
        # Hide the "0" labels
        for i, j in zip(*np.nonzero(matrix_with_hands == 0)):
            disp.text_[i, j].set_visible(False)

        # ax.set_title("MediaPipe + XGBoost — MegaDataset Users Test Set ")
        f2.set_tight_layout(True)

    return f1, f2

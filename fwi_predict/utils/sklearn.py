
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, TransformerMixin
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# Probably want to change this so that it doesn't use the plot() function and more closely matches original CalibrationDisplay interface.
class MulticlassCalibrationDisplay:
  """
  A display for visualizing the calibration curves of a multiclass classification model.
  """

  def __init__(self, prob_true, prob_pred, classes):
    """Initialize the CalibrationDisplay.

    Parameters:
    - prob_true (list of arrays): List of true probabilities for each class.
    - prob_pred (list of arrays): List of predicted probabilities for each class.
    - classes (list): List of class labels.
    """
    self.prob_true = prob_true
    self.prob_pred = prob_pred
    self.classes = classes


  @classmethod
  def from_estimator(cls, estimator, X, y, encoder=None, n_bins=10, strategy='uniform'):
    """Create a CalibrationDisplay from an estimator.

    Parameters:
    - estimator: A fitted classifier with a `predict_proba` method.
    - X: Feature matrix.
    - y: True labels (already encoded).
    - encoder: Label encoder to get class names. If None, uses estimator.classes_.
    - n_bins: Number of bins for the calibration curve.
    - strategy: Strategy to define the bins ('uniform' or 'quantile').

    Returns:
    - CalibrationDisplay instance.
    """
    if not is_classifier(estimator):
        raise ValueError("The estimator should be a classifier.")

    y_pred_prob = estimator.predict_proba(X)
    classes = encoder.classes_ if encoder is not None else estimator.classes_
    return cls.from_predictions(y, y_pred_prob, classes, n_bins=n_bins, strategy=strategy)


  @classmethod
  def from_predictions(cls, y_true, y_pred_prob, classes, n_bins=10, strategy='uniform'):
    """Create a CalibrationDisplay from true labels and predicted probabilities.

    Parameters:
    - y_true: True labels (already encoded).
    - y_pred_prob: Predicted probabilities.
    - classes: List of class labels.
    - n_bins: Number of bins for the calibration curve.
    - strategy: Strategy to define the bins ('uniform' or 'quantile').

    Returns:
    - CalibrationDisplay instance.
    """
    if len(classes) <= 2:
      raise ValueError(
        "For binary classification, use sklearn.calibration.CalibrationDisplay instead. "
        "This class is intended for multiclass calibration."
      )

    y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
    prob_true = []
    prob_pred = []

    for i in range(len(classes)):
      true_class = y_true_binarized[:, i]
      pred_class = y_pred_prob[:, i]

      prob_true_class, prob_pred_class = calibration_curve(
          true_class, pred_class, n_bins=n_bins, strategy=strategy
      )
      prob_true.append(prob_true_class)
      prob_pred.append(prob_pred_class)

    return cls(prob_true, prob_pred, classes)


  def plot(self, ax=None, figsize=(10, 8)):
    """Plot calibration curves for each class.

    Parameters:
    - ax: A matplotlib axis object. If None, a new figure and axis are created.
    - figsize (tuple): Size of the plot figure (ignored if ax is provided).
    """
    no_ax_passed = False
    if ax is None:
      fig, ax = plt.subplots(figsize=figsize)
      no_ax_passed = True

    for i, class_label in enumerate(self.classes):
      ax.plot(
        self.prob_pred[i], self.prob_true[i], marker='o', label=f"{class_label}", linewidth=1.5
      )

    ax.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    ax.set_title("Calibration Curves for Multiclass Classification")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="best")
    ax.grid()

    if no_ax_passed:
      plt.show()


# Define diurnal detrend transform
class DiurnalDetrend(BaseEstimator, TransformerMixin):
    """Detrend data by subtracting morning/evening means."""
    def fit(self, X, y=None):
        # Store feature names from X
        self.feature_names_in_ = np.asarray(X.columns.tolist())
        
        # Calculate means for morning/evening
        df = pd.DataFrame({'y': y, 'morning': X['morning']})
        self.morning_mean_ = df[df['morning']]['y'].mean()
        self.evening_mean_ = df[~df['morning']]['y'].mean()
        return self

    def transform(self, X, y=None):
        if y is not None:
            y_detrended = y.copy()
            y_detrended[X['morning']] -= self.morning_mean_
            y_detrended[~X['morning']] -= self.evening_mean_
            return y_detrended
        return X
    
    def inverse_transform(self, X, y):
        y_retrended = y.copy()
        y_retrended[X['morning']] += self.morning_mean_
        y_retrended[~X['morning']] += self.evening_mean_
        return y_retrended
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : list of str or None, default=None
            Input feature names. If None, then feature_names_in_ is used.
            
        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names.
        """
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features)
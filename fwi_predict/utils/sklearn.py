
import matplotlib.pyplot as plt
from sklearn.base import is_classifier
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
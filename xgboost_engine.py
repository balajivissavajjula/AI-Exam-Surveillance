"""
AI Exam Surveillance
XGBoost Behavior Classification Engine

Input:
    Exact 37-feature representation

Output:
    Normal / Cheating
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config


logger = logging.getLogger(__name__)


class XGBoostEngine:
    """
    Loads the trained XGBoost model and preprocessing pipeline
    and performs Normal vs Cheating classification.
    """

    def __init__(
        self,
        model_path=None,
        preprocessor_path=None,
    ):
        self.model_path = Path(
            model_path or config.XGBOOST_MODEL
        )

        self.preprocessor_path = Path(
            preprocessor_path or config.XGBOOST_PREPROCESSOR
        )

        self.model = None
        self.preprocessor = None
        self.loaded = False

        self.feature_names = config.FEATURE_NAMES.copy()

        if len(self.feature_names) != 37:
            raise ValueError(
                f"Expected exactly 37 features, "
                f"found {len(self.feature_names)}"
            )

        self.load()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(self):
        """
        Load XGBoost model and preprocessing artifact.
        """

        if not self.model_path.is_file():
            logger.warning(
                "XGBoost model not found: %s",
                self.model_path,
            )
            return False

        if not self.preprocessor_path.is_file():
            logger.warning(
                "XGBoost preprocessor not found: %s",
                self.preprocessor_path,
            )
            return False

        try:
            self.preprocessor = joblib.load(
                self.preprocessor_path
            )

            self.model = joblib.load(
                self.model_path
            )

            self.loaded = True

            logger.info(
                "XGBoost engine loaded successfully"
            )

            logger.info(
                "Model: %s",
                self.model_path,
            )

            logger.info(
                "Preprocessor: %s",
                self.preprocessor_path,
            )

            return True

        except Exception as exc:
            self.loaded = False

            logger.exception(
                "Failed to load XGBoost artifacts: %s",
                exc,
            )

            return False

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    def validate_features(self, features):
        """
        Validate the exact 37-feature representation.
        """

        if features is None:
            raise ValueError(
                "Features cannot be None"
            )

        if isinstance(features, dict):

            missing = [
                name
                for name in self.feature_names
                if name not in features
            ]

            if missing:
                raise ValueError(
                    "Missing features: "
                    + ", ".join(missing)
                )

            return {
                name: features[name]
                for name in self.feature_names
            }

        if isinstance(features, pd.DataFrame):

            missing = [
                name
                for name in self.feature_names
                if name not in features.columns
            ]

            if missing:
                raise ValueError(
                    "Missing features: "
                    + ", ".join(missing)
                )

            return features[
                self.feature_names
            ]

        if isinstance(features, (list, tuple, np.ndarray)):

            if len(features) != 37:
                raise ValueError(
                    f"Expected 37 features, "
                    f"received {len(features)}"
                )

            return list(features)

        raise TypeError(
            "Features must be a dict, DataFrame, "
            "list, tuple, or numpy array"
        )

    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    def create_feature_vector(self, features):
        """
        Convert a 37-feature dictionary/list into a
        single-row DataFrame using the exact feature order.
        """

        validated = self.validate_features(features)

        if isinstance(validated, dict):

            row = [
                validated[name]
                for name in self.feature_names
            ]

        elif isinstance(validated, pd.DataFrame):

            return validated.copy()

        else:
            row = list(validated)

        return pd.DataFrame(
            [row],
            columns=self.feature_names,
        )

    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(self, features):
        """
        Apply the saved preprocessing pipeline.
        """

        dataframe = self.create_feature_vector(
            features
        )

        if self.preprocessor is None:
            raise RuntimeError(
                "XGBoost preprocessor is not loaded"
            )

        try:
            transformed = self.preprocessor.transform(
                dataframe
            )

            return transformed

        except Exception as exc:
            logger.exception(
                "Feature preprocessing failed: %s",
                exc,
            )
            raise

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, features):
        """
        Perform Normal vs Cheating prediction.

        Returns:
            {
                prediction,
                label,
                confidence,
                normal_probability,
                cheating_probability,
                reasons
            }
        """

        if not self.loaded:
            raise RuntimeError(
                "XGBoost engine is not loaded"
            )

        transformed = self.preprocess(
            features
        )

        try:
            prediction = self.model.predict(
                transformed
            )

            label = int(
                np.asarray(prediction).reshape(-1)[0]
            )

        except Exception as exc:
            logger.exception(
                "XGBoost prediction failed: %s",
                exc,
            )
            raise

        normal_probability = 0.0
        cheating_probability = 0.0

        if hasattr(
            self.model,
            "predict_proba"
        ):
            probabilities = self.model.predict_proba(
                transformed
            )

            probabilities = np.asarray(
                probabilities
            )[0]

            if len(probabilities) >= 2:
                normal_probability = float(
                    probabilities[0]
                )

                cheating_probability = float(
                    probabilities[1]
                )

        else:

            if label == config.XGBOOST_CHEATING_LABEL:
                cheating_probability = 1.0
            else:
                normal_probability = 1.0

        if label == config.XGBOOST_CHEATING_LABEL:

            prediction_name = "Cheating"

            confidence = cheating_probability

        else:

            prediction_name = "Normal"

            confidence = normal_probability

        return {
            "prediction": prediction_name,
            "label": label,
            "confidence": float(confidence),
            "normal_probability": float(
                normal_probability
            ),
            "cheating_probability": float(
                cheating_probability
            ),
            "features_ready": True,
            "reasons": [],
        }

    # ========================================================
    # SAFE PREDICTION
    # ========================================================

    def predict_safe(self, features):
        """
        Prediction wrapper that never crashes the
        surveillance pipeline.
        """

        try:

            if not self.loaded:

                return {
                    "prediction": "Normal",
                    "label": 0,
                    "confidence": 0.0,
                    "normal_probability": 0.0,
                    "cheating_probability": 0.0,
                    "features_ready": False,
                    "reasons": [
                        "XGBoost model not loaded"
                    ],
                }

            return self.predict(features)

        except Exception as exc:

            logger.exception(
                "Safe XGBoost prediction failed: %s",
                exc,
            )

            return {
                "prediction": "Normal",
                "label": 0,
                "confidence": 0.0,
                "normal_probability": 0.0,
                "cheating_probability": 0.0,
                "features_ready": False,
                "reasons": [
                    f"XGBoost error: {exc}"
                ],
            }

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def get_feature_importance(self):
        """
        Return feature importance when supported by the
        loaded XGBoost model.
        """

        if self.model is None:
            return {}

        try:

            if not hasattr(
                self.model,
                "feature_importances_",
            ):
                return {}

            values = np.asarray(
                self.model.feature_importances_
            )

            names = self.feature_names

            # The trained preprocessor may expand
            # 37 raw features into processed features.
            if len(values) != len(names):

                return {
                    f"processed_feature_{i}": float(
                        value
                    )
                    for i, value in enumerate(values)
                }

            importance = {
                name: float(value)
                for name, value in zip(
                    names,
                    values,
                )
            }

            return dict(
                sorted(
                    importance.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        except Exception as exc:

            logger.warning(
                "Unable to read feature importance: %s",
                exc,
            )

            return {}

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):
        """
        Return engine status for dashboard/API.
        """

        return {
            "loaded": bool(self.loaded),
            "model_path": str(
                self.model_path
            ),
            "preprocessor_path": str(
                self.preprocessor_path
            ),
            "feature_count": len(
                self.feature_names
            ),
            "feature_names": self.feature_names.copy(),
            "classes": config.BEHAVIOR_CLASSES.copy(),
        }


# ============================================================
# GLOBAL ENGINE
# ============================================================

_engine = None


def get_xgboost_engine():
    """
    Return the shared XGBoost engine instance.
    """

    global _engine

    if _engine is None:
        _engine = XGBoostEngine()

    return _engine


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def predict_behavior(features):
    """
    Convenience function for detector.py.
    """

    engine = get_xgboost_engine()

    return engine.predict_safe(
        features
    )


def validate_37_features(features):
    """
    Validate the exact 37-feature input.
    """

    engine = get_xgboost_engine()

    return engine.validate_features(
        features
    )


def get_feature_names():
    """
    Return the exact 37 feature names.
    """

    return config.FEATURE_NAMES.copy()


def get_engine_status():
    """
    Return XGBoost engine status.
    """

    return get_xgboost_engine().get_status()


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    print("=" * 60)
    print("XGBoost Behavior Classification Engine")
    print("=" * 60)

    engine = XGBoostEngine()

    print(
        "Loaded:",
        engine.loaded,
    )

    print(
        "Feature count:",
        len(engine.feature_names),
    )

    print(
        "Model:",
        engine.model_path,
    )

    print(
        "Preprocessor:",
        engine.preprocessor_path,
    )

    print("=" * 60)
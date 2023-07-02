import logging
import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Xray(object):
    """Class xray"""

    def __init__(self, models_dir="./models/XrayModel_self/1"):
        self.loaded = False
        logging.info("Load model here...")
        self._models_dir = models_dir

    def load(self):
        self._xray_model = tf.keras.models.load_model(f"{self._models_dir}")
        self.loaded = True
        logging.info("Model has been loaded and initialized...")

    def predict(self, X, feature_names=None):
        """Predict Method"""
        if not self.loaded:
            logging.info("Not loaded yet.")
            self.load()
        logging.info("Model loaded.")
        probs = self._xray_model.predict(X)
        return probs

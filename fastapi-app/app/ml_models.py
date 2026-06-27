#app/ml_models.py
import joblib
import numpy as np
import os

# Paths to models
MODEL_DIR = 'models'
KMEANS_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
PCA_PATH = os.path.join(MODEL_DIR, 'pca_model.pkl')
TOPIC_CENTROIDS_PATH = os.path.join(MODEL_DIR, 'topic_centroids.pkl')
CENTROIDS_PCA_PATH = os.path.join(MODEL_DIR, 'centroids_pca.pkl')
CENTROIDS_NORM_PATH = os.path.join(MODEL_DIR, 'centroids_norm.pkl')

class MLModels:
    def __init__(self):
        self.kmeans = None
        self.pca = None
        self.topic_centroids = None
        self.centroids_pca = None
        self.centroids_norm = None

    def load_models(self):
        try:
            if os.path.exists(KMEANS_PATH):
                self.kmeans = joblib.load(KMEANS_PATH)
                self.pca = joblib.load(PCA_PATH)
                self.topic_centroids = joblib.load(TOPIC_CENTROIDS_PATH)
                self.centroids_pca = joblib.load(CENTROIDS_PCA_PATH)
                self.centroids_norm = joblib.load(CENTROIDS_NORM_PATH)
                print("ML Models loaded successfully.")
            else:
                print("ML Models not found. Please run kmeans.py first.")
        except Exception as e:
            print(f"Error loading ML models: {e}")

ml_models = MLModels()

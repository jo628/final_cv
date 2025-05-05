
# Script to save feature scaler parameters
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create a synthetic scaler based on reasonable feature ranges
def create_and_save_scaler():
    # Generate synthetic features with reasonable ranges for waste classification
    synthetic_features = []
    for _ in range(100):
        # Deep features typically have values in this range after ImageNet pretraining
        deep_features = np.random.randn(1280) * 2.0
        # Color features often normalized to [0,1]
        color_features = np.random.rand(58)
        # Texture and material features vary
        texture_features = np.random.randn(250) * 0.5
        material_features = np.random.randn(10) * 0.5
        
        # Combine all features - make sure dimensions match your actual features
        features = np.concatenate([deep_features, color_features, texture_features, material_features])
        synthetic_features.append(features)
    
    # Fit scaler on synthetic data
    scaler = StandardScaler()
    scaler.fit(np.array(synthetic_features))
    
    # Save the scaler
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Feature scaler saved to feature_scaler.pkl")

if __name__ == "__main__":
    create_and_save_scaler()

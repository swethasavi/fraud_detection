# module2_model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

class ModelDevelopmentModule:
    """
    Module 2: Machine Learning Model Development
    This module trains and evaluates fraud detection models
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['amount', 'hour', 'day_of_week', 'is_foreign', 'prev_transactions']
        
    def load_and_prepare_data(self, data_file):
        """Load and prepare training data"""
        print("Loading data...")
        
        # Generate synthetic data if file doesn't exist
        try:
            df = pd.read_csv(data_file)
        except:
            # Create synthetic data
            np.random.seed(42)
            n_samples = 10000
            
            data = {
                'amount': np.random.uniform(1, 2000, n_samples),
                'hour': np.random.randint(0, 24, n_samples),
                'day_of_week': np.random.randint(0, 7, n_samples),
                'is_foreign': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'prev_transactions': np.random.randint(0, 50, n_samples),
                'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
            }
            df = pd.DataFrame(data)
            
            # Make fraud patterns realistic
            fraud_mask = df['is_fraud'] == 1
            df.loc[fraud_mask, 'amount'] = np.random.uniform(500, 2000, fraud_mask.sum())
            df.loc[fraud_mask, 'hour'] = np.random.choice([1,2,3,4,22,23], fraud_mask.sum())
            
            df.to_csv(data_file, index=False)
            print(f"Created synthetic data with {len(df)} transactions")
        
        print(f"Data shape: {df.shape}")
        print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        
        return df
    
    def train_model(self, df):
        """Train the fraud detection model"""
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Logistic Regression - simple and interpretable)
        print("\nTraining Logistic Regression model...")
        self.model = LogisticRegression(class_weight='balanced', random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate False Positive Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"False Positives: {fp} transactions")
        
        # Feature importance (coefficients for Logistic Regression)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', ascending=False)
        
        print("\nFeature Importance (coefficients):")
        print(feature_importance)
        
        # Save test data for later
        test_results = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_prob
        })
        test_results.to_csv('test_results.csv', index=False)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': false_positive_rate,
            'false_positives': fp
        }
    
    def save_model(self, model_path='fraud_model.pkl', scaler_path='scaler.pkl'):
        """Save trained model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

# Usage example
if __name__ == "__main__":
    # Initialize module
    ml_module = ModelDevelopmentModule()
    
    # Load and prepare data
    df = ml_module.load_and_prepare_data('training_data.csv')
    
    # Train model
    metrics = ml_module.train_model(df)
    
    # Save model
    ml_module.save_model()
    
    print("\n✅ Module 2 completed successfully!")
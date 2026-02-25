# module3_api_development.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from bank_simulator import bank  # Import the bank simulator

app = Flask(__name__)

class FraudDetectionAPI:
    """
    Module 3: Real-time API Development
    This module creates a REST API for fraud detection
    """
    
    def __init__(self):
        # Load model and scaler
        print("Loading fraud detection model...")
        try:
            self.model = joblib.load('fraud_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.feature_columns = ['amount', 'hour', 'day_of_week', 'is_foreign', 'prev_transactions']
            print("✅ Model loaded successfully!")
        except:
            print("⚠️ Model files not found. Please run module2 first.")
            self.model = None
            self.scaler = None
        
        # Simple token for demo
        self.valid_token = "demo_token_123"
        print("✅ API initialized!")
        print("✅ Connected to Bank Simulator")
    
    def validate_token(self, auth_header):
        """Validate API token"""
        if not auth_header:
            return False
        
        try:
            token = auth_header.split(' ')[1]
            return token == self.valid_token
        except:
            return False
    
    def preprocess_transaction(self, data):
        """Clean and format transaction data"""
        processed = {}
        
        # Handle amount
        amount = data.get('amount', 0)
        if isinstance(amount, str):
            amount = amount.replace('$', '').replace(',', '')
        processed['amount'] = float(amount)
        
        # Handle timestamp
        timestamp = data.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                processed['hour'] = dt.hour
                processed['day_of_week'] = dt.weekday()
            except:
                processed['hour'] = datetime.now().hour
                processed['day_of_week'] = datetime.now().weekday()
        else:
            processed['hour'] = int(data.get('hour', datetime.now().hour))
            processed['day_of_week'] = int(data.get('day_of_week', datetime.now().weekday()))
        
        # Handle location
        country = data.get('country', 'US')
        processed['is_foreign'] = 1 if country != 'US' else 0
        processed['country'] = country  # Store for display
        
        # Previous transactions
        processed['prev_transactions'] = int(data.get('prev_transactions', 0))
        
        # Card number (for freezing)
        processed['card_number'] = data.get('card_number', '4111111111111111')
        processed['customer'] = data.get('customer', 'John Doe')
        
        return processed
    
    def predict(self, transaction_data):
        """Make fraud prediction"""
        if self.model is None:
            return 0, 0.0
        
        # Convert to DataFrame
        input_df = pd.DataFrame([transaction_data])
        
        # Ensure correct column order
        input_df = input_df[self.feature_columns]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        probability = self.model.predict_proba(input_scaled)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability

# Initialize API
api = FraudDetectionAPI()

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Fraud Detection API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/health': 'GET - Check API health',
            '/predict': 'POST - Analyze single transaction',
            '/dashboard': 'GET - View web dashboard',
            '/frozen-cards': 'GET - See all frozen cards'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': api.model is not None,
        'bank_connected': True
    })

@app.route('/frozen-cards', methods=['GET'])
def frozen_cards():
    """Show all frozen cards"""
    frozen = bank.get_freeze_history()
    return jsonify({
        'frozen_cards': frozen,
        'count': len(frozen)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single transaction prediction with automatic card freezing"""
    
    try:
        # Get transaction data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get card and customer info (if not provided, use defaults)
        card_number = data.get('card_number', '4111111111111111')
        customer = data.get('customer', 'John Doe')
        
        # First, check if card is already frozen
        card_status = bank.check_card_status(card_number)
        if card_status == 'FROZEN':
            print(f"❌ Card ****{card_number[-4:]} is already frozen - transaction rejected")
            return jsonify({
                'transaction_id': data.get('transaction_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'decision': 'REJECTED',
                'action': '❌ CARD ALREADY FROZEN',
                'fraud_probability': 1.0,
                'message': 'This card has been frozen due to previous suspicious activity'
            })
        
        print(f"\n🔍 Processing transaction: {data.get('transaction_id', 'unknown')}")
        print(f"Card: ****{card_number[-4:]}")
        print(f"Customer: {customer}")
        print(f"Amount: ${data.get('amount', 0)}")
        print(f"Country: {data.get('country', 'US')}")
        
        # Preprocess data
        processed = api.preprocess_transaction(data)
        
        # Make prediction
        prediction, probability = api.predict(processed)
        
        # Determine action and automatically freeze card if fraud detected
        if prediction == 1:
            decision = "FRAUD"
            action = "❌ CARD FROZEN"
            emoji = "🚨"
            
            # AUTOMATICALLY FREEZE THE CARD!
            transaction_details = {
                'amount': data.get('amount'),
                'country': data.get('country'),
                'time': datetime.now().strftime("%H:%M:%S"),
                'transaction_id': data.get('transaction_id')
            }
            
            bank.freeze_card(
                card_number=card_number,
                reason=f"Suspicious transaction: ${data.get('amount')} from {data.get('country')}",
                transaction_details=transaction_details
            )
            
        else:
            decision = "LEGITIMATE"
            action = "✅ APPROVED"
            emoji = "✅"
        
        print(f"{emoji} Decision: {decision}")
        print(f"📊 Fraud Probability: {probability:.2%}")
        
        # Get current frozen cards count
        frozen_count = len(bank.get_freeze_history())
        
        # Prepare response
        response = {
            'transaction_id': data.get('transaction_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'action': action,
            'fraud_probability': float(probability),
            'confidence': float(probability if prediction == 1 else 1 - probability),
            'card_last4': card_number[-4:],
            'customer': customer,
            'frozen_cards_total': frozen_count
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Web dashboard for testing"""
    return render_template('dashboard.html')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 FRAUD DETECTION API WITH AUTO-FREEZE")
    print("="*50)
    print("📝 To test the API:")
    print("   - Dashboard: http://localhost:5000/dashboard")
    print("   - Health check: http://localhost:5000/health")
    print("   - View frozen cards: http://localhost:5000/frozen-cards")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
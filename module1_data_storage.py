# module1_data_storage.py
import boto3
import json
import csv
from datetime import datetime

class DataStorageModule:
    """
    Module 1: Data Collection and Storage
    This module handles storing transaction data in DynamoDB
    """
    
    def __init__(self):
        # Connect to DynamoDB
        self.dynamodb = boto3.resource('dynamodb')
        self.table_name = 'CreditCardTransactions'
        self.create_table()
    
    def create_table(self):
        """Create DynamoDB table for transactions"""
        try:
            # Check if table exists
            existing_tables = self.dynamodb.tables.all()
            table_names = [table.name for table in existing_tables]
            
            if self.table_name not in table_names:
                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {'AttributeName': 'TransactionID', 'KeyType': 'HASH'},  # Partition key
                        {'AttributeName': 'Timestamp', 'KeyType': 'RANGE'}      # Sort key
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'TransactionID', 'AttributeType': 'S'},
                        {'AttributeName': 'Timestamp', 'AttributeType': 'N'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                print(f"Creating table {self.table_name}...")
                table.wait_until_exists()
                print("Table created successfully!")
            else:
                print(f"Table {self.table_name} already exists")
                
        except Exception as e:
            print(f"Error creating table: {e}")
    
    def import_transactions_from_csv(self, csv_file):
        """Import transaction data from CSV file"""
        table = self.dynamodb.Table(self.table_name)
        
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                transaction = {
                    'TransactionID': row['transaction_id'],
                    'Timestamp': int(datetime.now().timestamp()),
                    'Amount': float(row['amount']),
                    'CardNumber': row.get('card_number', '****1234'),
                    'Location': row.get('location', 'Unknown'),
                    'Merchant': row.get('merchant', 'Unknown'),
                    'IsFraud': int(row.get('is_fraud', 0))
                }
                
                # Insert into DynamoDB
                table.put_item(Item=transaction)
                print(f"Added transaction: {transaction['TransactionID']}")
    
    def query_transactions(self, transaction_id=None, date_range=None):
        """Query transactions from database"""
        table = self.dynamodb.Table(self.table_name)
        
        if transaction_id:
            response = table.get_item(
                Key={'TransactionID': transaction_id, 'Timestamp': date_range}
            )
            return response.get('Item')
        else:
            # Scan for all transactions (limit for demo)
            response = table.scan(Limit=100)
            return response.get('Items', [])

# Usage example
if __name__ == "__main__":
    storage = DataStorageModule()
    
    # Create sample CSV data if not exists
    sample_data = """transaction_id,amount,location,merchant,is_fraud
txn_001,125.50,New York,Amazon,0
txn_002,4500.00,London,Apple Store,1
txn_003,35.99,Chicago,Starbucks,0
txn_004,890.00,Paris,LV,1"""
    
    with open('sample_transactions.csv', 'w') as f:
        f.write(sample_data)
    
    # Import data
    storage.import_transactions_from_csv('sample_transactions.csv')
    
    # Query data
    transactions = storage.query_transactions()
    print(f"Total transactions: {len(transactions)}")
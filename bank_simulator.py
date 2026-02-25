# bank_simulator.py
"""
Bank Simulator - Shows which cards are frozen
"""

class BankSimulator:
    """
    Simulates a bank that manages cards and freezes them when fraud is detected
    """
    
    def __init__(self):
        # Dictionary to store card status: {card_number: status}
        # status: 'ACTIVE' or 'FROZEN'
        self.cards = {}
        self.frozen_cards_log = []
        print("🏦 Bank Simulator Started")
        print("="*50)
    
    def register_card(self, card_number, customer_name):
        """Register a new card with the bank"""
        self.cards[card_number] = {
            'status': 'ACTIVE',
            'customer': customer_name,
            'frozen_at': None
        }
        print(f"✅ Card registered: ****{card_number[-4:]} ({customer_name}) - ACTIVE")
        return True
    
    def freeze_card(self, card_number, reason, transaction_details):
        """Automatically freeze a card when fraud is detected"""
        if card_number in self.cards:
            from datetime import datetime
            
            # Freeze the card
            self.cards[card_number]['status'] = 'FROZEN'
            freeze_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cards[card_number]['frozen_at'] = freeze_time
            
            # Log the freeze event
            freeze_event = {
                'card': card_number[-4:],  # Show only last 4 digits
                'customer': self.cards[card_number]['customer'],
                'time': freeze_time,
                'reason': reason,
                'transaction': transaction_details
            }
            self.frozen_cards_log.append(freeze_event)
            
            # Print alert
            print("\n" + "🔴"*25)
            print("🚨 CARD FROZEN AUTOMATICALLY!")
            print("🔴"*25)
            print(f"Card: ****{card_number[-4:]}")
            print(f"Customer: {self.cards[card_number]['customer']}")
            print(f"Time: {freeze_time}")
            print(f"Reason: {reason}")
            print(f"Transaction Amount: ${transaction_details.get('amount', 0)}")
            print(f"Location: {transaction_details.get('country', 'Unknown')}")
            print("🔴"*25 + "\n")
            
            return True
        return False
    
    def check_card_status(self, card_number):
        """Check if a card is active or frozen"""
        if card_number in self.cards:
            return self.cards[card_number]['status']
        return 'NOT_REGISTERED'
    
    def unfreeze_card(self, card_number):
        """Manually unfreeze a card (customer service action)"""
        if card_number in self.cards and self.cards[card_number]['status'] == 'FROZEN':
            self.cards[card_number]['status'] = 'ACTIVE'
            print(f"✅ Card ****{card_number[-4:]} unfrozen")
            return True
        return False
    
    def get_freeze_history(self):
        """Get history of all freeze events"""
        return self.frozen_cards_log

# Create a global bank instance with some test cards
bank = BankSimulator()

# Register some test cards
bank.register_card("4111111111111111", "John Doe")
bank.register_card("5555555555554444", "Jane Smith")
bank.register_card("378282246310005", "Bob Johnson")

print("\n✅ Bank simulator ready with 3 test cards")
print("   - John Doe (****1111)")
print("   - Jane Smith (****4444)") 
print("   - Bob Johnson (****0005)")
print("="*50)
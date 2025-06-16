from faker import Faker
import pandas as pd
import random

fake = Faker()

sensitive_fields = ['name', 'email', 'ssn', 'phone', 'address']
non_sensitive_fields = [
    'transaction_amount', 'product', 'date', 'product_category',
    'payment_method', 'store_location', 'customer_feedback', 
    'delivery_time_days', 'product_rating', 'quantity', 'brand'
]

product_categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Grocery', 'Toys', 'Furniture']
payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Cash', 'PayPal', 'Net Banking']
feedback_options = ['Great service!', 'Okay', 'Poor quality', 'Fast shipping', 'Will buy again', 'Not satisfied']
brands = ['Samsung', 'Apple', 'Nike', 'Sony', 'Ikea', 'LG', 'Adidas']

rows = 10000
data = []

for _ in range(rows):
    row = {
        'name': fake.name(),
        'email': fake.email(),
        'ssn': fake.ssn(),
        'phone': fake.phone_number(),
        'address': fake.address().replace("\n", ", "),
        'transaction_amount': round(fake.pyfloat(left_digits=4, right_digits=2, positive=True), 2),
        'product': fake.word(),
        'date': fake.date(),
        'product_category': random.choice(product_categories),
        'payment_method': random.choice(payment_methods),
        'store_location': fake.city() + ", " + fake.state(),
        'customer_feedback': random.choice(feedback_options),
        'delivery_time_days': random.randint(1, 10),
        'product_rating': round(random.uniform(1.0, 5.0), 1),
        'quantity': random.randint(1, 5),
        'brand': random.choice(brands),
    }

    data.append(row)

df = pd.DataFrame(data)
df.to_csv("sensitive_non_sensitive_dataset_full.csv", index=False)
print("✅ Dataset saved as 'sensitive_non_sensitive_dataset_full.csv'")

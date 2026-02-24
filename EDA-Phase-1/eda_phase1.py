import pandas as pd

print("Loading small datasets...")

customers = pd.read_csv("customers.csv")
accounts = pd.read_csv("accounts.csv")
linkage = pd.read_csv("customer_account_linkage.csv")
labels = pd.read_csv("train_labels.csv")

print("Customers shape:", customers.shape)
print("Accounts shape:", accounts.shape)
print("Linkage shape:", linkage.shape)
print("Labels shape:", labels.shape)

print("\nLabels columns:")
print(labels.columns)

print("\nFirst 5 rows of labels:")
print(labels.head())
print("\nFraud (Mule) Distribution:")
print(labels['is_mule'].value_counts())

print("\nFraud Percentage:")
print(labels['is_mule'].value_counts(normalize=True) * 100)
print("\nLoading transactions sample...")

transactions = pd.read_csv("transactions_part_0.csv")

print("Transactions shape:", transactions.shape)

print("\nTransactions columns:")
print(transactions.columns)

print("\nFirst 5 transactions:")
print(transactions.head())
print("\nMerging transactions with mule labels...")

transactions = transactions.merge(
    labels[['account_id', 'is_mule']],
    on='account_id',
    how='left'
)

# Replace NaN with 0 (accounts not in labels assumed non-mule)
transactions['is_mule'] = transactions['is_mule'].fillna(0)

print("\nAfter merge:")
print(transactions['is_mule'].value_counts())
print("\nTransaction Amount Summary by Mule Flag:")

print(transactions.groupby('is_mule')['amount'].describe())

print("\nTransaction Type vs Mule:")

print(pd.crosstab(transactions['txn_type'], transactions['is_mule']))

print("\nTransaction Count per Account:")

txn_count = transactions.groupby('account_id').size().reset_index(name='txn_count')

txn_with_flag = txn_count.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')

print(txn_with_flag.groupby('is_mule')['txn_count'].describe())
print("\nChannel vs Mule Distribution:")

channel_table = pd.crosstab(transactions['channel'], transactions['is_mule'])
print(channel_table)

print("\nChannel Percentage (Row-wise):")
print(channel_table.div(channel_table.sum(axis=1), axis=0) * 100)

transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'])

transactions['hour'] = transactions['transaction_timestamp'].dt.hour

print("\nAverage Hour of Transaction by Mule Flag:")
print(transactions.groupby('is_mule')['hour'].describe())

print("\nUnique Counterparties per Account:")

counterparty_count = transactions.groupby('account_id')['counterparty_id'].nunique().reset_index(name='unique_counterparties')

counterparty_with_flag = counterparty_count.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')

print(counterparty_with_flag.groupby('is_mule')['unique_counterparties'].describe())

print("\n================ ACCOUNT LEVEL FEATURE ENGINEERING ================")

# Aggregate transaction features per account
account_features = transactions.groupby("account_id").agg({
    "amount": ["count", "mean", "std", "min", "max", "sum"],
    "counterparty_id": "nunique",
    "hour": "mean"
}).reset_index()

# Flatten column names
account_features.columns = [
    "account_id",
    "txn_count",
    "avg_amount",
    "std_amount",
    "min_amount",
    "max_amount",
    "total_amount",
    "unique_counterparties",
    "avg_txn_hour"
]

print("\nAccount Features Shape:")
print(account_features.shape)

credit_debit = transactions.groupby(['account_id', 'txn_type']).size().unstack(fill_value=0)

credit_debit['credit_debit_ratio'] = (
    credit_debit.get('C', 0) / 
    (credit_debit.get('D', 0) + 1)
)

account_features = account_features.merge(
    credit_debit[['credit_debit_ratio']],
    on='account_id',
    how='left'
)
flow = transactions.groupby(['account_id', 'txn_type'])['amount'].sum().unstack(fill_value=0)

flow['net_flow'] = flow.get('C', 0) - flow.get('D', 0)

account_features = account_features.merge(
    flow[['net_flow']],
    on='account_id',
    how='left'
)
transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'])
transactions['transaction_date'] = transactions['transaction_timestamp'].dt.date

print("Adding Transaction Intensity Feature...")

txn_per_day = transactions.groupby(['account_id', 'transaction_date']).size()
txn_per_day = txn_per_day.groupby('account_id').mean().reset_index()

txn_per_day.columns = ['account_id', 'avg_txn_per_day']

account_features = account_features.merge(
    txn_per_day,
    on='account_id',
    how='left'
)
print("\nFirst 5 rows of Account Features:")
print(account_features.head())
print("\nMerging Account Features with Mule Labels...")

account_level = account_features.merge(
    labels[['account_id', 'is_mule']],
    on='account_id',
    how='left'
)

account_level['is_mule'] = account_level['is_mule'].fillna(0)
account_level = account_level.fillna(0)

print("\nAccount-Level Mule Distribution:")
print(account_level['is_mule'].value_counts()) 
print("\n================ ACCOUNT LEVEL COMPARISON ================")

comparison = account_level.groupby("is_mule").mean(numeric_only=True)
print(comparison)
print("\nTop 10 Accounts by Transaction Count:")
print(account_level.sort_values("txn_count", ascending=False).head(10))
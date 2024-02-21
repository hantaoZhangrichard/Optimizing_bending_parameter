import pandas as pd

# Load Excel file into a DataFrame
df = pd.read_excel('/Users/angela/Desktop/Book2.xlsx', sheet_name='Sheet1')

# print(df.columns.tolist()) 

column_to_check = df['Unnamed: 2']

# Create a set with all numbers from 1 to 1512
all_numbers = set(range(1, 1512))

# Check if all numbers are in the column
missing_numbers = all_numbers - set(column_to_check)

if not missing_numbers:
    print("All numbers from 1 to 1512 are in the column.")
else:
    print("Missing numbers in the column:", missing_numbers)

import pandas as pd
df = pd.read_csv('outputs/results/glims_retreat_assessment_validated.csv', nrows=2)
for i, c in enumerate(df.columns):
    print(f"{i}: '{c}'")

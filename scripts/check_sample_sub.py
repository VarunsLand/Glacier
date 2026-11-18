import pandas as pd
p = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/glims_verified_repeated_observations.csv"
df = pd.read_csv(p, low_memory=False)
sub = df[(df['glac_id'] == 'G005937E44988N') & (df['src_date'].str.startswith('2003-08-13'))]
print(sub.to_string())

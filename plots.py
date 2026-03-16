import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("/home/oso/code/spar_deception/scrapper/greaterwrong_split_512_classified.csv")


prob_cols = [
    'class_0_prob', 'class_1_prob', 'class_2_prob', 
    'class_3_prob', 'class_4_prob', 'class_5_prob'
]

# 2. Determine the class with the maximum probability for each row
# This creates a new column with the name of the column that has the highest value
df['max_prob_class'] = df[prob_cols].idxmax(axis=1)

# Optional: Clean up labels (e.g., 'class_0_prob' -> 'Class 0')
df['max_prob_class'] = df['max_prob_class'].str.replace('_prob', '').str.replace('_', ' ').str.title()
class_2_rows = df[df['max_prob_class'] == 'Class 5']

# Display the rows
print(class_2_rows)
# 3. Plotting the distribution
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a count plot of the winners
ax = sns.countplot(
    data=df, 
    x='max_prob_class', 
    palette='viridis', 
    order=sorted(df['max_prob_class'].unique())
)

# Add numeric counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

plt.title('Distribution of Labels (Based on Max Probability)', fontsize=15)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Save the plot
plt.savefig('class_distribution.png')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Load the data from the Excel file
file_path = r'Freq.ERM.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path, engine='openpyxl')

# Extract and sort the relevant columns by 'Freq.ERM'
df_sorted = df.sort_values(by='Freq.ERM', ascending=True)
algorithms = df_sorted['Algorithm']
frequencies = df_sorted['Freq.ERM']

# Define hatch patterns for black-and-white differentiation
hatch_patterns = ['/', '\\', '|', '-', '+', 'x',  '//', '\\\\', '||', '--', '++', 'xx','.','o', 'O', '*']

# Plotting the histogram
plt.figure(figsize=(10, 3.5))
# colors: https://matplotlib.org/stable/gallery/color/named_colors.html
bars = plt.bar(algorithms, frequencies, color='lightgrey', edgecolor='black')

# Apply hatch patterns
for bar, hatch in zip(bars, hatch_patterns):
    bar.set_hatch(hatch)

# Adding labels
plt.xlabel('Algorithm', fontsize=13)
plt.ylabel('$Freq_{ERM}$', fontsize=13)

# Rotate x labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)

# Adding value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=12)

# Making the upper and right edges invisible
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Making the layout tight and saving the figure
plt.tight_layout()
# plt.savefig('histogram.png', dpi=300)  # Saving the figure as a high-resolution PNG file
plt.savefig('histogram.pdf')  # Saving the figure as a high-resolution PNG file
plt.show()
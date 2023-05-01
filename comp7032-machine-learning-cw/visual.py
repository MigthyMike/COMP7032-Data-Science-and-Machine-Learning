import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Hotel Reservations.csv')
value_counts = df["booking_status"].value_counts()

plt.figure(figsize=(6, 6))
ax = value_counts.plot(kind='bar')  # Assign the bar chart to a variable called 'ax'
plt.title(f'Distribution of booking_status')
plt.xlabel("booking status")
plt.ylabel('Frequency')
plt.xticks(rotation='horizontal')

# Add the total number of occurrences as text above each bar
for i, v in enumerate(value_counts):
    ax.annotate(str(v), xy=(i, v), xycoords='data', ha='center', va='bottom')

plt.show()
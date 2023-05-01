from matplotlib import pyplot as plt

data = [
(1, 0.8240),
(2, 0.8635),
(3, 0.9001),
(4, 0.9030),
(5, 0.9062),
(6, 0.9066),
(7, 0.9070),
    ]


x_values, y_values = zip(*data)

plt.plot(x_values, y_values, marker='o')

plt.xlabel('n_components')
plt.ylabel('Accuracy')
plt.title('PCA Accuracy vs. n_components')

plt.xticks(x_values)

plt.show()

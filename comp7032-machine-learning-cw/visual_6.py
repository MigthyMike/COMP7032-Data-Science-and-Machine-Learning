from matplotlib import pyplot as plt

data = [
(1, 0.7580),
(2, 0.8117),
(3, 0.8278),
(4, 0.8340),
(5, 0.8653),
(6, 0.8637),
(7, 0.8635),
(8, 0.8669),
(9, 0.8706),
(10, 0.8714),
(11, 0.8715),
(12, 0.8726),
(13, 0.8759),
(14, 0.8748),
(15, 0.8764),
(16, 0.8786),
(17, 0.8773),
    ]


x_values, y_values = zip(*data)

plt.plot(x_values, y_values, marker='o')

plt.xlabel('n_components')
plt.ylabel('Accuracy')
plt.title('PCA Accuracy vs. n_components')

plt.xticks(x_values)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([43, 154, 80, 60, 67, 91, 34, 50, 60, 59])

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, color="blue")
plt.plot(x, y_pred, color="red")
plt.title("2026.10.05 14:00")
plt.xlabel("Rank")
plt.ylabel("Likes (x1,000)")
plt.savefig("melon_regression.png")
plt.show()

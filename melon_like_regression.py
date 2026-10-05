import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "Malgun Gothic"

titles = ["퇴사할게여", "LOVE ATTACK", "갑자기", "Deja Vu", "Pretty Girl",
          "REDRED", "BAD", "만찬가", "It's Me", "LEMONADE"]
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([43, 154, 80, 60, 67, 91, 34, 50, 60, 59])

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, color="blue")
plt.plot(x, y_pred, color="red")
for i, t in enumerate(titles):
    plt.annotate(t, (x[i, 0], y[i]), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8)
plt.title("2026.10.05 14:00")
plt.xlabel("Rank")
plt.ylabel("Likes (x1,000)")
plt.xticks(range(1, 11))
plt.xlim(0.3, 10.7)
plt.ylim(25, 170)
plt.savefig("melon_regression.png")
plt.show()

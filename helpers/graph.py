# 8 Rank: 0.214251s

# 4 Ranks: 0.208991s

# 2 Ranks: 0.234318s

# 1 Ranks: 0.194184s

weak = [0.194184, 0.234318, 0.208991, 0.214251]

# 8 Rank: 0.311105s

# 4 Ranks: 0.449133s

# 2 Ranks: 0.728292s

# 1 Ranks: 0.922766s

strong = [0.922766, 0.728292, 0.449133, 0.311105]

ranks = [1, 2, 4, 8]

speedup_weak = [weak[0] / weak[i] for i in range(len(weak))]
speedup_strong = [strong[0] / strong[i] for i in range(len(strong))]

import matplotlib.pyplot as plt

plt.plot(ranks, speedup_weak, label="Weak Scaling", marker="o")
plt.plot(ranks, speedup_strong, label="Strong Scaling", marker="o")
plt.xlabel("Number of Ranks")
plt.ylabel("Speedup")
plt.title("Weak vs Strong Scaling")
plt.xticks(ranks)
plt.grid()
plt.legend()
plt.show()

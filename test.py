import numpy as np
# Example array with values
oa = np.array([0.2, 0.3, 0.499, 0.6, 0.7])

ma = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

diffs = abs(oa - ma)

rs = oa ** diffs

print(diffs)
print(rs)

print(oa + rs)



import numpy as np
import matplotlib.pyplot as plt

blocks = []
current = []

with open("tcf_bb/bb.xvg") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith(('#', '@')):
            continue

        if line.startswith('&'):
            if current:
                blocks.append(np.array(current, float))
                current = []
        else:
            current.append(line.split())

# append last block
if current:
    blocks.append(np.array(current, float))

# extract correlation column from each block
corr = np.column_stack([b[:, 1] for b in blocks])

print(corr.shape)
time_lags = np.arange(corr.shape[0])
plt.plot(time_lags, corr)
plt.title("tau_c Estimate from GMX ACF")
plt.xlabel("Lag Time (frames)")
plt.ylabel("ACF")

plt.savefig("bb_tcfs.png", dpi=300)

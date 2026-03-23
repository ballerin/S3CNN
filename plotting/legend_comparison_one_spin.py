import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color='#FFBB28', lw=2, linestyle=':', label='sCNN single spin (train)'),
    Line2D([0], [0], color='#FFBB28', lw=2, linestyle='-', label='sCNN single spin (val)'),
    Line2D([0], [0], color='#2ca02c', lw=2, linestyle=':', label='sCNN dual spin (train)'),
    Line2D([0], [0], color='#2ca02c', lw=2, linestyle='-', label='sCNN dual spin (val)'),
    Line2D([0], [0], color='#7570b3', lw=2, linestyle=':', label='GsCNN single spin (train)'),
    Line2D([0], [0], color='#7570b3', lw=2, linestyle='-', label='GsCNN single spin (val)'),
    Line2D([0], [0], color='#d62728', lw=2, linestyle=':', label='GsCNN dual spin (train)'),
    Line2D([0], [0], color='#d62728', lw=2, linestyle='-', label='GsCNN dual spin (val)'),
]

fig, ax = plt.subplots()

legend = ax.legend(
    handles=handles,
    ncol=4,
    loc='center',
    frameon=True,
    handletextpad=0.4,
    columnspacing=0.8,
    labelspacing=0.4,
)

ax.axis('off')

plt.savefig("figures/legend_comparison_one_spin.png", bbox_inches='tight')
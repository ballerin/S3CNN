import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color='#D81B60', lw=2, linestyle=':', label='swsCNN single spin (train)'),
    Line2D([0], [0], color='#D81B60', lw=2, linestyle='-', label='swsCNN single spin (val)'),
    Line2D([0], [0], color='#1E88E5', lw=2, linestyle=':', label='swsCNN dual spin (train)'),
    Line2D([0], [0], color='#1E88E5', lw=2, linestyle='-', label='swsCNN dual spin (val)'),
    Line2D([0], [0], color='#FFC107', lw=2, linestyle=':', label='S3CNN single spin (train)'),
    Line2D([0], [0], color='#FFC107', lw=2, linestyle='-', label='S3CNN single spin (val)'),
    Line2D([0], [0], color='#004D40', lw=2, linestyle=':', label='S3CNN dual spin (train)'),
    Line2D([0], [0], color='#004D40', lw=2, linestyle='-', label='S3CNN dual spin (val)'),
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color='#1b9e77', lw=2, linestyle='-', label='CNN NR/NR'),
    Line2D([0], [0], color='#1b9e77', lw=2, linestyle='--', label='CNN NR/R'),
    Line2D([0], [0], color='#1b9e77', lw=2, linestyle=':', label='CNN R/R'),
    Line2D([0], [0], color='#d95f02', lw=2, linestyle='-', label='swsCNN NR/NR'),
    Line2D([0], [0], color='#d95f02', lw=2, linestyle='--', label='swsCNN NR/R'),
    Line2D([0], [0], color='#d95f02', lw=2, linestyle=':', label='swsCNN R/R'),
    Line2D([0], [0], color='#7570b3', lw=2, linestyle='-', label='S3CNN NR/NR'),
    Line2D([0], [0], color='#7570b3', lw=2, linestyle='--', label='S3CNN NR/R'),
    Line2D([0], [0], color='#7570b3', lw=2, linestyle=':', label='S3CNN R/R'),
]

fig, ax = plt.subplots()

legend = ax.legend(
    handles=handles,
    ncol=3,
    loc='center',
    frameon=True,
    handletextpad=0.4,
    columnspacing=0.8,
    labelspacing=0.4,
)

ax.axis('off')

plt.savefig("figures/legend_dataset_size.png", bbox_inches='tight')
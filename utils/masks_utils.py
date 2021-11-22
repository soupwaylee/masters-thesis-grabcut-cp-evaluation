import numpy as np

import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


kromp_cmap = sns.color_palette("PRGn", 5)


def idx_str_to_np_array(x):
    indices = [int(s) for s in x.split(',')]
    mask_array = np.empty((384, 512), dtype=np.uint8).ravel()
    mask_array.fill(0)
    mask_array[indices] = 1
    return mask_array.reshape((384, 512))


def get_iou_for_mask_row(row, gt):
    image_id = row['image_id']
    i = np.logical_and(row['mask_arr'], gt[image_id])
    u = np.logical_or(row['mask_arr'], gt[image_id])
    return np.sum(i) / np.sum(u)


def compute_mean_iou_for_predictions(predictions, groundtruth):
    i = np.logical_and(predictions, groundtruth)
    u = np.logical_or(predictions, groundtruth)
    return np.sum(i) / np.sum(u)


def plot_iou_box_per_kromp(kromp_mask_df):
    sns.set_style({
        'xtick.top': False,
        'ytick.right': False,
        'ytick.left': False,
    })

    f, ax1 = plt.subplots(figsize=(6, 4))

    PROPS = {
        'boxprops':{'facecolor':'none'},
        'medianprops':{'color':'#ca0020', 'linewidth': 3},
        'capprops':{'linewidth': 2.5}
    }

    sns.boxplot(y='image_id', x='iou', data=kromp_mask_df, showfliers = False, width=.3, linewidth=0.75, whis=1.8, **PROPS)

    sns.stripplot(y='image_id', x='iou', data=kromp_mask_df, size=6, color=".2", alpha=.4, jitter=0.35)

    ax1.xaxis.grid(True)
    ax1.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
    ax1.set(xlabel="IoU")
    ax1.set(ylabel="")
    ax1.set_xlim([0.3, 1.01])
    sns.despine(offset=10, trim=True)
    ax1.spines['left'].set_visible(False)

    plt.savefig(f"plots_png/kromp_annotated_iou.png", bbox_inches='tight')
    plt.savefig(f"plots_png_higher_res/kromp_annotated_iou.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"plots_pdf/kromp_annotated_iou.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_iou_cdfs(ious_cdfs_df):
    sns.set_style({
        'xtick.top': False,
        'ytick.right': False,
        'ytick.left': True,
    })

    f, ax = plt.subplots(figsize=(6, 3))

    marker_style={
        'markeredgecolor': 'dimgrey'
    }

    sns.lineplot(
        data=ious_cdfs_df,
        x='xs',
        y='ps',
        dashes=False,
        hue='image_id',
        style='image_id',
        ci=None,
        markers=['o', 'o', 'o', 'o', 'o'],
        linewidth=3,
        markersize=8,
        palette='PRGn',
        ax=ax, **marker_style)

    ax.set_xlim([0.3, 1.0])

    l = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labelspacing=0.8, frameon=False)

    ax.set(ylabel="CDF")
    ax.set(xlabel="IoU")
    sns.despine(offset=10, trim=True)

    plt.savefig(f"plots_png/kromp_annotated_iou_CDF.png", bbox_inches='tight')
    plt.savefig(f"plots_png_higher_res/kromp_annotated_iou_CDF.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"plots_pdf/kromp_annotated_iou_CDF.pdf", format='pdf', bbox_inches='tight')

    plt.show()


def plot_iou_to_duration(kromp_mask_df):
    sns.set_style({
        'xtick.top': False,
        'ytick.right': False,
        'ytick.left': True,
    })

    f, ax2 = plt.subplots(figsize=(7, 5))

    g = sns.scatterplot(x="interaction_duration", y="iou", hue="image_id", size="total_submissions",
                        sizes=(50, 1200), alpha=.5, palette=kromp_cmap, edgecolor='dimgrey', linewidth=2,
                        data=kromp_mask_df, ax=ax2)

    ax2.set(xlabel="Interaction Duration (s)")
    ax2.set(ylabel="IoU")

    ax2.set_ylim([0.4, 1.01])
    ax2.set_xlim([0, 600])
    ax2.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

    sns.despine(offset=10, trim=True)

    l = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labelspacing=0.6, frameon=False)

    plt.setp(l.texts, family='DejaVu Sans')

    t1, t2 = l.get_texts()[0], l.get_texts()[6]
    t1._fontproperties = t2._fontproperties.copy()

    t1.set_text('Image ID:')
    t1.set_weight('bold')
    t2.set_text('Segmentation\nattempts:')
    t2.set_weight('bold')

    plt.savefig(f"plots_png/kromp_iou_to_duration.png", bbox_inches='tight')
    plt.savefig(f"plots_png_higher_res/kromp_iou_to_duration.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"plots_pdf/kromp_iou_to_duration.pdf", format='pdf', bbox_inches='tight')

    plt.show()


def plot_iou_to_area(kromp_mask_df):
    sns.set_style({
        'xtick.top': False,
        'ytick.right': False,
    })

    f, ax3 = plt.subplots(figsize=(7, 5))


    g = sns.scatterplot(x="annotated_pixels_percentage", y="iou", hue="image_id", size="total_submissions",
                        sizes=(50, 1200), alpha=.5, palette=kromp_cmap, edgecolor='dimgrey', linewidth=2,
                        data=kromp_mask_df, ax=ax3)

    ax3.set(xlabel="Annotated pixel area (%)")
    ax3.set(ylabel="IoU")

    ax3.set_ylim([0.4, 1.01])
    ax3.set_xlim([-0.015, 0.252])

    new_labels = [i * 5 for i in range(6)]
    ax3.set_xticklabels(new_labels)
    ax3.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

    sns.despine(offset=10, trim=True)

    l = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labelspacing=0.6, frameon=False)

    plt.setp(l.texts, family='DejaVu Sans')

    t1, t2 = l.get_texts()[0], l.get_texts()[6]
    t1._fontproperties = t2._fontproperties.copy()

    t1.set_text('Image ID:')
    t1.set_weight('bold')

    t2.set_text('Segmentation\nattempts:')
    t2.set_weight('bold')

    plt.savefig(f"plots_png/kromp_iou_to_area.png", bbox_inches='tight')
    plt.savefig(f"plots_png_higher_res/kromp_iou_to_area.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"plots_pdf/kromp_iou_to_area.pdf", format='pdf', bbox_inches='tight')

    plt.show()
"""
Created on Fri Mar 19 14:38:01 2021

@author: mozhenling
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
def fontsizes(base_fontsize, legend=False):
    """
    Calculate font sizes for axis labels, title, and ticks based on a base fontsize.

    Parameters:
    base_fontsize : int or float
        The base font size.

    Returns:
    tuple
        A tuple containing (axis_label_fontsize, title_fontsize, tick_fontsize).
    """
    axis_label_fontsize = base_fontsize
    legend_fontsize = base_fontsize
    title_fontsize = base_fontsize * 1.2
    tick_fontsize = base_fontsize
    if legend:
        return axis_label_fontsize, title_fontsize, tick_fontsize, legend_fontsize
    else:
        return axis_label_fontsize, title_fontsize, tick_fontsize

def smooth_data(y, method=None, **kwargs):
    """
    Apply a smoothing method to the data.

    Parameters:
    y : list or array
        The y-data to be smoothed.
    method : str, optional
        The smoothing method to apply. Options: 'moving_average', 'savgol', 'gaussian'.
        If None, no smoothing is applied.
    kwargs : dict
        Additional parameters for the smoothing methods.

    Returns:
    smoothed_y : array
        The smoothed y-data.
    """
    if method in ['avg','moving_average']:
        window_size = kwargs.get('window_size', 3)
        if window_size is None:
            return y
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        return np.convolve(y, np.ones(window_size) / window_size, mode='same')
    elif method in ['svg','savgol']:
        window_size = kwargs.get('window_size', 5)
        if window_size is None:
            return y
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(y, window_size, polyorder)
    elif method in ['gsn','gaussian']:
        sigma = kwargs.get('sigma', 1)
        if sigma is None:
            return y
        return gaussian_filter1d(y, sigma)
    else:
        return y  # No smoothing applied


def multi_y_curves(x, y_data, xy_labels, y0_color=None, title=None, save_dir=None, format='png',
                   y_extra_colors=('tab:blue', 'tab:green', 'tab:purple', 'tab:orange'),
                   smoothing=None, smoothing_params=None,
                   fontsize=12, figsize=(4, 2), dpi=300, alpha=0.5):
    """
    Plot multiple curves with different y-axes and optional smoothing, with transparency.

    Parameters:
    x : list or array
        The shared x-axis data.
    y_data : list of lists/arrays
        A list containing y-data for each curve. Each element corresponds to one y-axis.
    xy_labels : list of tuples
        A list of tuples with each tuple specifying the label for the x-axis and corresponding y-axis.
        Format: [(x_label, y_label1), (x_label, y_label2), ...]
    save_dir : str, optional
        Directory to save the plot. If None, the plot is not saved.
    format : str, optional
        Format to save the plot (e.g., 'png', 'jpg').
    y_extra_colors : tuple, optional
        Tuple of colors for the different curves.
    smoothing : str, optional
        Smoothing method to apply to the data. Options: 'moving_average', 'savgol', 'gaussian', or None.
    smoothing_params : dict or list of dicts, optional
        Parameters for the smoothing methods. If a list is provided, each dictionary corresponds to a curve.
        If a single dictionary is provided, it applies to all curves.
    fontsize : int, optional
        Base font size for labels, title, and ticks.
    figsize : tuple, optional
        Size of the figure in inches.
    dpi : int, optional
        Dots per inch for the figure.
    alpha : float, optional
        Transparency level for the curves (default is 0.7). Range: 0.0 (fully transparent) to 1.0 (fully opaque).
    """
    assert len(y_data) - 1 <= len(y_extra_colors), "Please adjust the number of color tags to be no smaller than the number of curves"

    # Ensure smoothing_params is a list of dictionaries, one for each y_data
    if smoothing_params is None:
        smoothing_params = [{}] * len(y_data)
    elif isinstance(smoothing_params, dict):
        smoothing_params = [smoothing_params] * len(y_data)
    elif len(smoothing_params) != len(y_data):
        raise ValueError("smoothing_params must have the same length as y_data or be a single dictionary.")

    # Get font sizes
    axis_label_fontsize, title_fontsize, tick_fontsize = fontsizes(fontsize)

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot the first y-axis
    y0_color = 'tab:red' if y0_color is None else y0_color
    ax1.set_xlabel(xy_labels[0][0], fontsize=axis_label_fontsize)
    ax1.set_ylabel(xy_labels[0][1], color=y0_color, fontsize=axis_label_fontsize)
    smoothed_y = smooth_data(y_data[0], method=smoothing, **smoothing_params[0])
    ax1.plot(x, smoothed_y, color=y0_color)  # Add alpha transparency
    ax1.tick_params(axis='y', labelcolor=y0_color, labelsize=tick_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize)

    if len(y_data) > 1:
        # Create additional y-axes
        axes = [ax1]  # List to keep track of axes for further customization
        for i, y in enumerate(y_data[1:], start=1):
            ax = ax1.twinx()  # Create a new y-axis
            ax.spines['right'].set_position(('outward', 60 * (i - 1)))  # Offset each new y-axis
            color = y_extra_colors[i % len(y_extra_colors)]
            ax.set_ylabel(xy_labels[i][1], color=color, fontsize=axis_label_fontsize)
            smoothed_y = smooth_data(y, method=smoothing, **smoothing_params[i])
            ax.plot(x, smoothed_y, color=color, alpha=alpha)  # Add alpha transparency
            ax.tick_params(axis='y', labelcolor=color, labelsize=tick_fontsize)
            axes.append(ax)

    # Set the title with the appropriate fontsize
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
        # Adjust layout to avoid overlap
    else:
        fig.tight_layout()  # Leave space for the title

    # Save the plot if a directory is provided
    if save_dir is not None:
        plt.savefig(save_dir, format=format)

    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_names,title=None,
                          save_dir=None, format='png',fontsize=10, figsize=(3, 2),dpi=300):
    tick_marks = np.array(range(len(label_names))) + 0.5
    axis_label_fontsize, title_fontsize, tick_fontsize = fontsizes(fontsize)
    def _plot_matrix(cm, title, cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # title='Normalized Confusion Matrix'
        if title is not None:
            plt.title(title, fontsize=title_fontsize)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=tick_fontsize)
        xlocations = np.array(range(len(label_names)))
        plt.xticks(xlocations, label_names, fontsize=tick_fontsize)
        plt.yticks(xlocations, label_names, fontsize=tick_fontsize)
        plt.ylabel('True label', fontsize=axis_label_fontsize)
        plt.xlabel('Predicted label', fontsize=axis_label_fontsize)

    # Use sklearn's confusion_matrix function
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=label_names)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    plt.figure(figsize=figsize, dpi=dpi)
    ind_array = np.arange(len(label_names))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01 and c <= 0.6:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=tick_fontsize, va='center', ha='center')
        if c > 0.6:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=tick_fontsize, va='center', ha='center')

    # Offset the ticks
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    _plot_matrix(cm_normalized, title=title, cmap=plt.cm.Blues)

    # Show or save the confusion matrix
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()


def plotDesity(train_dict, test_dict, perplexity=30, n_iter=1000, random_state=42,
            title='Label', fontsize = 10, xlabel='T-SNE Component 1', ylabel='Density',
           save_dir = None, format = 'png', figsize = (14, 7), dpi = 300, non_text = False):
    """Compare the densities of t-sne component 1 of training and test domains for each label"""
    axis_label_fontsize, title_fontsize,tick_fontsize = fontsizes(fontsize)
    legend_fontsize = fontsize

    train_features = train_dict['x']
    train_labels = train_dict['y']
    test_features = test_dict['x']
    test_labels = test_dict['y']

    # Combine train and test features for TSNE
    combined_features = np.vstack((train_features, test_features))
    combined_labels = np.hstack((train_labels, test_labels))
    domains = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))

    # Run TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(combined_features)

    # Get unique labels
    unique_labels = np.unique(combined_labels)

    # Plot distributions of each class of features for both domains
    fig, axs = plt.subplots(len(unique_labels), 1, figsize=(figsize[0], len(unique_labels) * 3), dpi = dpi , sharex=True)
    for i, label in enumerate(unique_labels):
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'train'), 0], ax=axs[i], color='blue', label='Train', shade=True)
        sns.kdeplot(tsne_results[(combined_labels == label) & (domains == 'test'), 0], ax=axs[i], color='red', label='Test', shade=True)
        axs[i].set_title(title+f' {label}', fontsize=title_fontsize)
        axs[i].set_xlabel(xlabel, fontsize=axis_label_fontsize)
        axs[i].set_ylabel(ylabel, fontsize=axis_label_fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[i].legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()

def plotTSNE(train_dict, test_dict, perplexity=30, n_iter=1000, random_state=42,
             axis_tight=True, title=None, fontsize = 14, xlabel='T-SNE Component 1', ylabel='T-SNE Component 2',
           save_dir = None, format = 'png', figsize = (8, 6), dpi = 300, non_text = False, marker_size=100, is_grid=False):
    """visualize features from training and test domains on 2d t-sne plots"""
    axis_label_fontsize, title_fontsize, tick_fontsize, legend_fontsize = fontsizes(fontsize,legend=True)

    train_features = train_dict['x']
    train_labels = train_dict['y']
    test_features = test_dict['x']
    test_labels = test_dict['y']

    # Combine train and test features for TSNE
    combined_features = np.vstack((train_features, test_features))
    combined_labels = np.hstack((train_labels, test_labels))
    domains = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))

    # Run TSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(combined_features)

    # Get unique labels
    unique_labels = np.unique(combined_labels)
    markers = ['o','*', 's', 'D', '^', 'x', 'p',  '<', '>', 'h']  # A list of markers
    domain_colors = {'train': 'blue', 'test': 'red'}  # Colors for domains

    plt.figure(figsize=figsize, dpi=dpi)

    # Plot TSNE results with different colors for domains and markers for labels
    for i, label in enumerate(unique_labels):
        for domain in ['train', 'test']:
            indices = (combined_labels == label) & (domains == domain)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        marker=markers[i % len(markers)],
                        color=domain_colors[domain],
                        label=f'{domain} label {label}' if domain == 'train' else f'{domain} label {label}',
                        alpha=0.7, s=marker_size)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_label_fontsize)
    plt.ylabel(ylabel, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best',  fontsize=legend_fontsize)
    plt.grid(is_grid)
    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()


def hparams_plot1D(x, val_avg, val_std,test_avg, test_std, axis_tight=True, title=None, fontsize = 10,
                   xlabel='log10(β)', ylabel='Accuracy', save_dir = None, format = 'png', figsize = (8, 6), dpi = 300, non_text = False):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size=fontsize)
    # Plot val_avg with error bars
    plt.errorbar(x, val_avg, yerr=val_std, fmt='-o', color='blue', label='val_avg+/-std', capsize=5)

    # Plot test_avg with error bars
    plt.errorbar(x, test_avg, yerr=test_std, fmt='-s', color='red', label='test_avg+/-std', capsize=5)

    # Adding labels and title
    plt.xlabel(xlabel, fontsize=fontsize+1)
    plt.ylabel(ylabel, fontsize=fontsize+1)
    if title is not None:
        plt.title(title, fontsize=fontsize+2)

    plt.tight_layout()
    plt.legend(fontsize=fontsize + 0.5)

    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()

def hparams_plot3D(X, Y, Z, markoptimal=True, axis_tight=True, title=None, fontsize = 10, xlabel='x', ylabel='y', zlabel='z',
           save_dir = None, format = 'png', figsize = (8, 6), dpi = 300, non_text = False, elev = 30, azim=60,
                   x_decimals=2, y_decimals=1, z_decimals=1, opt_displace=0.028):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.rc('font', size=fontsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=('viridis'), zorder = 1)  # 'cool' looks good # 'viridis'
    # Create the format string based on the decimals variable
    # format_decimals = f'%.{decimals}f'
    # Format the ticks to show the specified number of decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{x_decimals}f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{y_decimals}f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter(f'%.{z_decimals}f'))
    # Set tick label fontsize
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='z', which='major', labelsize=fontsize)

    if axis_tight:
        ax.set_xlim(min(X.flatten()), max(X.flatten()))
        ax.set_ylim(min(Y.flatten()), max(Y.flatten()))
        ax.set_zlim(min(Z.flatten()), max(Z.flatten()))

    if non_text:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axes.zaxis.set_visible(False)
    else:
        ax.set_xlabel(xlabel, fontsize = fontsize + 1)
        ax.set_ylabel(ylabel, fontsize = fontsize + 1)
        ax.set_zlabel(zlabel, fontsize = fontsize + 1)
        ax.set_title(title, fontsize = fontsize + 2)

    if markoptimal:
        # Find the optimal point (max z value in this case)
        optimal_idx = np.argmax(Z)
        X_opt = X.flatten()[optimal_idx]
        Y_opt = Y.flatten()[optimal_idx]
        Z_opt = Z.flatten()[optimal_idx]
        Z_opt = Z_opt + opt_displace * (max(Z.flatten()) - min(Z.flatten()))
        # Annotate the optimal point with its coordinates
        ax.text(X_opt, Y_opt, Z_opt,
                f'({X_opt:.{x_decimals}f}, {Y_opt:.{y_decimals}f}, {Z_opt:.{z_decimals}f})',
                color='black',zorder=100, fontsize=fontsize)
        # Mark the optimal point
        ax.scatter(X_opt, Y_opt, Z_opt, color='red', s=50, label='Optimal Point', zorder=100)
        # Use quiver to draw an arrow pointing to the optimal point
        # arrow_length = 10  # Length of the arrow
        # ax.quiver(X_opt, Y_opt, Z_opt + arrow_length,
        #           0, 0, -arrow_length,
        #           color='r', arrow_length_ratio=0.1, linewidth=2, zorder=100)

    #-- save the image
    plt.tight_layout()
    ax.legend(fontsize=fontsize+0.5)

    # Adjust the vantage point
    # https: // matplotlib.org / stable / api / toolkits / mplot3d / view_angles.html
    ax.view_init(elev=elev, azim=azim)

    if save_dir is not None:
        plt.savefig(save_dir, format=format)
    plt.show()



def get_X_AND_Y(X_min, X_max, Y_min, Y_max, reso=0.01, step=None):
    """
    reso: normalized resolution for the image
    num: number of points
    step: un-normalized resolution for the image

    num = 1 / reso
    reso = 1 / num
    step = (x_max-x_min )/ num
    step = reso * (x_max - x_min)

    """
    if reso is not None and step is None:
        step_used = reso * (X_max - X_min)
    elif step is not None and reso is None:
        step_used = step
    elif reso is not None and step is not None:
        raise ValueError("Choose reso (resolution in percentage) or step?")
    else:
        raise ValueError('reso and step should not be both None!')
    X = np.arange(X_min, X_max, step_used)
    Y = np.arange(Y_min, Y_max, step_used)
    X, Y = np.meshgrid(X, Y)
    return (X, Y)


# -- for debugging
def Rastrigin(X=None, Y=None, objMin=True, is2Show=False, X_min=-5.52, X_max=5.12, Y_min=-5.12, Y_max=5.12, **kwargs):
    A = 10
    if is2Show:
        X, Y = get_X_AND_Y(X_min, X_max, Y_min, Y_max, **kwargs)
        Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
        return (
            X, Y, Z, 100, 'Rastrigin function-3D')
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    if objMin:
        return Z
    return -Z


if __name__ == '__main__':
    X_min = -1
    X_max = 3
    Y_min = -1
    Y_max = 3

    num = 50
    reso = 1/num

    X, Y, Z, z_max, title = Rastrigin(X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max, reso= reso, is2Show=True)  # Schwefel, Rastrigin
    # plot3D(X, Y, Z, z_max, title)



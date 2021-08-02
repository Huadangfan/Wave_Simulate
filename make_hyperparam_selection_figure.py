import sys
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
from fwi import (Fwi, shuffle_shots, extract_datasets)
from setup_seam import setup_seam

def make_hyperparam_selection_figure():
    """Create and plot data comparing the final cost function value/loss
    when using different hyperparameters.
    """
    np.random.seed(0)

    model_init, dx, dt, train_dataset, dev_dataset, propagator = \
            _prepare_datasets()

    batch_sizes, learning_rates, losses = _make_figure_data(model_init,
                                                            dx, dt,
                                                            train_dataset,
                                                            dev_dataset,
                                                            propagator)

    _save_figure_data(batch_sizes, learning_rates, losses)

    _make_plots(batch_sizes, learning_rates, losses)


def _prepare_datasets():
    """Load the initial model, datasets, and other relevant values."""
    seam_model_path = sys.argv[1]
    seam = setup_seam(seam_model_path)
    model_init = seam['model_init']
    dx = seam['dx']
    dt = seam['dt']
    sources = seam['sources']
    sources_x = seam['sources_x']
    receivers_x = seam['receivers_x']
    propagator = seam['propagator']
    num_train_shots = seam['num_train_shots']

    seam_data_path = sys.argv[2]
    receivers = np.load(seam_data_path)

    dataset = sources, sources_x, receivers, receivers_x

    dataset = shuffle_shots(dataset)
    train_dataset, dev_dataset = extract_datasets(dataset, num_train_shots)

    return model_init, dx, dt, train_dataset, dev_dataset, propagator


def _make_figure_data(model_init, dx, dt, train_dataset, dev_dataset,
                      propagator):
    """Run optimization for different hyperparameter combinations and record
    the final cost function values.

    Returns:
        batch_sizes: A list of the batch size used in each trial
        learning_rates: A list of the learning rate used in each trial
        losses: A list of the final cost function values obtained in each trial
    """
    num_trials = 20
    batch_sizes = np.random.randint(low=2, high=6, size=num_trials)
    learning_rates = np.round(10**(np.random.random(num_trials)*4), 2)

    num_train_shots = train_dataset[0].shape[1]

    losses = []
    for trial_idx in range(num_trials):
        tf.reset_default_graph()
        num_steps = (num_train_shots // 2) // batch_sizes[trial_idx]
        optimizer = tf.train.AdamOptimizer(learning_rates[trial_idx])
        print(trial_idx, batch_sizes[trial_idx], learning_rates[trial_idx],
              num_steps)
        fwi = Fwi(model_init, dx, dt, train_dataset, dev_dataset, propagator,
                  optimizer=optimizer,
                  batch_size=batch_sizes[trial_idx])
        _, loss = fwi.train(num_steps, print_interval=num_steps - 1)
        losses.append(loss[1][-1][1])

    return batch_sizes, learning_rates, losses


def _save_figure_data(batch_sizes, learning_rates, losses):
    """Save the data that is needed to generate the plots so that they can be
    recreated without running the computations again, if necessary.
    """
    np.save('hyperparam_selection_batch_sizes', batch_sizes)
    np.save('hyperparam_selection_learning_rates', learning_rates)
    np.save('hyperparam_selection_losses', losses)


def _make_plots(batch_sizes, learning_rates, losses):
    """Create a plot of the final cost function values and save as an EPS file.
    """
    _, ax = plt.subplots(figsize=(5.4, 3.3))
    plt.style.use('grayscale')
    plt.rc({'font.size': 8})
    plt.scatter(x=batch_sizes, y=learning_rates, c=losses, edgecolors='black',
                vmax=500)
    plt.colorbar(label='Final loss')
    plt.xlabel('Batch size')
    plt.ylabel('Learning rate')
    ax.xaxis.set_major_locator(pltticker.MaxNLocator(integer=True))
    plt.title('Best learning rate: 45.4, batch size: 2')
    plt.tight_layout(pad=0)
    plt.savefig('hyperparam_selection.eps')


if __name__ == '__main__':
    make_hyperparam_selection_figure()

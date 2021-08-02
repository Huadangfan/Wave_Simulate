"""Test fwi.py
"""
import tensorflow as tf
import numpy as np
from fwi import (Fwi, _prepare_batch, _entire_dataset_loss_and_gradient)
from wavelets import ricker
from gen_data import gen_data
from forward1d import forward1d
from forward2d import forward2d

def test_gradient_1d(dx=5, dt=0.0001):
    """Compare the gradient of the cost function w.r.t. the wave speed
    produced by automatic differentiation, the adjoint-state method, and
    finite differences, on a 1D model."""
    np.random.seed(0)
    tf.reset_default_graph()
    nx = 50
    model_true = ((np.random.random(nx)-0.5)*200 + 1500).astype(np.float32)
    model_init = model_true + ((np.random.random(nx)-0.5)*50).astype(np.float32)
    nt = int(3 * nx * dx / 1500 / dt)
    propagator = forward1d
    sources_x = np.ones([1, 1, 1], np.int)
    receivers_x = sources_x.copy()
    sources = ricker(25, nt, dt, 0.05).reshape([-1, 1, 1])
    receivers = gen_data(model_true, dx, dt, sources, sources_x, receivers_x,
                         propagator)
    dataset = sources, sources_x, receivers, receivers_x

    # automatic differentiation
    fwi0 = Fwi(model_init, dx, dt, dataset, None, propagator,
               save_gradient=True)
    feed_dict = _prepare_batch(0, dataset, model_init.shape, 1,
                               fwi0.batch_placeholders)
    auto_gradient = fwi0.sess.run(fwi0.gradient, feed_dict=feed_dict)

    # adjoint gradient
    tf.reset_default_graph()
    fwi1 = Fwi(model_init, dx, dt, dataset, None, propagator, autodiff=False)
    feed_dict = _prepare_batch(0, dataset, model_init.shape, 1,
                               fwi1.batch_placeholders)
    adjoint_gradient = fwi1.sess.run(fwi1.gradient, feed_dict=feed_dict)

    loss_derivative = np.zeros(nx, np.float64)
    change_amp = 20
    for change_x in range(nx):
        new_loss = []
        for idx in range(2):
            tf.reset_default_graph()
            model_changed = model_init.copy()
            model_changed[change_x] += (-1)**idx * change_amp
            fwi2 = Fwi(model_changed, dx, dt, dataset, None, propagator)
            feed_dict = _prepare_batch(0, dataset, model_changed.shape,
                                       1, fwi2.batch_placeholders)
            new_loss.append(fwi2.sess.run(fwi2.loss, feed_dict=feed_dict))
        loss_derivative[change_x] = (new_loss[0] - new_loss[1])/(2*change_amp)

    #return auto_gradient[0][0], adjoint_gradient, loss_derivative
    assert np.allclose(auto_gradient[0][0], loss_derivative, atol=0.00015)
    assert np.allclose(adjoint_gradient, loss_derivative, atol=0.00015)


def test_gradient_2d(dx=5, dt=0.0001):
    """Compare the gradient of the cost function w.r.t. the wave speed
    produced by automatic differentiation, the adjoint-state method, and
    finite differences, on a 2D model.

    To reduce the computational cost, the finite difference gradient is only
    calculated on a coarse grid.
    """
    np.random.seed(1)
    tf.reset_default_graph()
    nz = 21
    nx = 21
    sample_freq = 4 # grid cell interval between f.d. gradient computations
    model_true = ((np.random.random([nz, nx])-0.5)*200
                  + 1500).astype(np.float32)
    model_init = (model_true
                  + ((np.random.random([nz, nx])-0.5)*50).astype(np.float32))
    nt = int(3 * np.sqrt(nz**2 + nx**2) * dx / 1500 / dt)
    propagator = forward2d
    num_sources = 2
    num_receivers = nx-2
    sources_x = np.ones([num_sources, 1, 2], np.int)
    sources_x[0, 0, 1] = nx // 3
    sources_x[1, 0, 1] = 2 * nx // 3
    receivers_x = np.ones([1, num_receivers, 2], np.int)
    receivers_x[0, :, 1] = np.arange(1, nx-1)
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    sources = ricker(25, nt, dt, 0.05).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    receivers = gen_data(model_true, dx, dt, sources, sources_x, receivers_x,
                         propagator)
    dataset = sources, sources_x, receivers, receivers_x
    batch_size = num_sources

    # automatic differentiation
    fwi0 = Fwi(model_init, dx, dt, dataset, None, propagator,
               batch_size=batch_size, save_gradient=True)
    feed_dict = _prepare_batch(0, dataset, model_init.shape,
                               batch_size, fwi0.batch_placeholders)
    auto_gradient = fwi0.sess.run(fwi0.gradient, feed_dict=feed_dict)

    # adjoint gradient
    tf.reset_default_graph()
    fwi1 = Fwi(model_init, dx, dt, dataset, None, propagator,
               batch_size=batch_size, autodiff=False)
    feed_dict = _prepare_batch(0, dataset, model_init.shape,
                               batch_size, fwi1.batch_placeholders)
    adjoint_gradient = fwi1.sess.run(fwi1.gradient, feed_dict=feed_dict)

    loss_derivative = np.zeros([nz // sample_freq + 1, nx // sample_freq + 1],
                               np.float32)
    change_amp = 20
    for change_z in range(0, nz, sample_freq):
        for change_x in range(0, nx, sample_freq):
            new_loss = []
            for idx in range(2):
                tf.reset_default_graph()
                model_changed = model_init.copy()
                model_changed[change_z, change_x] += (-1)**idx * change_amp
                fwi2 = Fwi(model_changed, dx, dt, dataset, None, propagator,
                           batch_size=batch_size)
                feed_dict = _prepare_batch(0, dataset,
                                           model_changed.shape,
                                           batch_size, fwi2.batch_placeholders)
                new_loss.append(fwi2.sess.run(fwi2.loss, feed_dict=feed_dict))
            loss_derivative[int(change_z/sample_freq),
                            int(change_x/sample_freq)] = \
                    (new_loss[0] - new_loss[1])/(2*change_amp)

    #return auto_gradient[0][0], adjoint_gradient, loss_derivative
    assert np.allclose(auto_gradient[0][0][::4, ::4], loss_derivative,
                       atol=4e-7)
    assert np.allclose(adjoint_gradient[::4, ::4], loss_derivative, atol=4e-7)


def test_entire_dataset(dx=5, dt=0.0001):
    """Check that the loss and gradient calculated on the entire dataset
    (by _entire_dataset_loss_and_gradient) when using several batches
    is the same as when using a single batch.
    """
    np.random.seed(2)
    tf.reset_default_graph()
    nz = 21
    nx = 21
    model_true = ((np.random.random([nz, nx])-0.5)*200
                  + 1500).astype(np.float32)
    model_init = (model_true
                  + ((np.random.random([nz, nx])-0.5)*50).astype(np.float32))
    nt = int(3 * np.sqrt(nz**2 + nx**2) * dx / 1500 / dt)
    propagator = forward2d
    num_sources = 6
    num_receivers = nx-2
    sources_x = np.ones([num_sources, 1, 2], np.int)
    sources_x[:, 0, 1] = np.linspace(1, nx-1, 6, dtype=np.int)
    receivers_x = np.ones([1, num_receivers, 2], np.int)
    receivers_x[0, :, 1] = np.arange(1, nx-1)
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])
    sources = ricker(25, nt, dt, 0.05).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    receivers = gen_data(model_true, dx, dt, sources, sources_x, receivers_x,
                         propagator)
    dataset = sources, sources_x, receivers, receivers_x

    # full dataset in one batch
    fwi0 = Fwi(model_init, dx, dt, dataset, None, propagator,
               save_gradient=True, batch_size=num_sources)
    feed_dict = _prepare_batch(0, dataset, model_init.shape,
                               fwi0.batch_size, fwi0.batch_placeholders)
    loss0, gradient0 = fwi0.sess.run([fwi0.loss, fwi0.gradient],
                                     feed_dict=feed_dict)

    # multiple batches
    tf.reset_default_graph()
    batch_size = 2
    fwi1 = Fwi(model_init, dx, dt, dataset, None, propagator,
               save_gradient=True, batch_size=batch_size)
    loss1, gradient1 = \
            _entire_dataset_loss_and_gradient(model_init, fwi1.model, dataset,
                                              None, batch_size,
                                              fwi1.batch_placeholders,
                                              fwi1.gradient, fwi1.loss,
                                              fwi1.sess, model_init.shape,
                                              None)

    #return loss0, loss1, gradient0, gradient1
    assert np.isclose(loss0, loss1, atol=5e-10)
    assert np.allclose(gradient0[0][0], gradient1.reshape([nz, nx]),
                       atol=2.25e-12)

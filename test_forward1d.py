import tensorflow as tf
import numpy as np
from wavelets import ricker
from forward1d import forward1d
from gen_data import gen_data

def green(x0, x1, dx, dt, t, v, v0, f):
    """Use the 1D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    y = np.sum(f[:np.maximum(0, int((t - np.abs(x1-x0)/v)/dt))])*dt*dx*v0/2
    return y


def test_direct_1d(v=1500, freq=25, dx=5, dt=0.0001, nx=80):
    """Create a constant model, and the expected waveform at point,
    and compare with forward propagated wave.
    """
    model = v * np.ones([nx], dtype=np.float32)

    num_sources = 1
    num_receivers = 1

    nt = int(2*nx*dx/v/dt)
    sources = ricker(freq, nt, dt, 0.05).reshape([-1, 1, 1])
    sources = np.tile(sources, [1, num_sources, 1])
    sources_x = np.zeros([num_sources, 1, 1], np.int)
    sources_x[:, :, 0] = 1
    receivers_x = np.zeros([1, num_receivers, 1], np.int)
    receivers_x[:, :, 0] = nx-5
    receivers_x = np.tile(receivers_x, [num_sources, 1, 1])

    expected = np.array([green(sources_x[0, 0, 0]*dx, receivers_x[0, 0, 0]*dx,
                               dx, dt, t, v, v, sources.ravel())
                         for t in np.arange(nt)*dt])

    actual = gen_data(model, dx, dt, sources, sources_x, receivers_x,
                      forward1d)

    #return expected, actual.ravel()
    assert np.allclose(expected, actual.ravel(), atol=1)


def test_reflect_1d(nsteps=None, v0=1500, v1=2500, freq=25, dx=5, dt=0.0006,
                    nx=100):
    """Create a model with one reflector, and the expected wavefield at one
    time, and compare with the forward propagated wave."""
    tf.reset_default_graph()
    reflector_x = int(nx/2)
    model = np.ones(nx, dtype=np.float32) * v0
    model[reflector_x:] = v1
    if nsteps is None:
        nsteps = np.ceil(0.2/dt).astype(np.int)
    source_x = int(.35 * nx)
    sources = ricker(freq, nsteps, dt, 0.05).reshape([-1, 1, 1])
    # create a new source shifted by the time to the reflector
    time_shift = np.round((reflector_x-source_x)*dx / v0 / dt).astype(np.int)
    shifted_source = np.pad(sources.reshape(-1),
                            (time_shift, 0), 'constant')
    expected = np.zeros([nx], dtype=np.float32)
    # reflection and transmission coefficients
    r = (v1 - v0) / (v1 + v0)
    t = 1 + r

    # direct wave
    expected[:reflector_x] = np.array([green(x*dx, source_x*dx, dx, dt,
                                             (nsteps+1)*dt, v0, v0,
                                             sources)
                                       for x in range(reflector_x)])
    # reflected wave
    expected[:reflector_x] += r*np.array([green(x*dx, (reflector_x-1)*dx, dx,
                                                dt, (nsteps+1)*dt, v0, v0,
                                                shifted_source)
                                          for x in range(reflector_x)])
    # transmitted wave
    expected[reflector_x:] = t*np.array([green(x*dx, reflector_x*dx, dx, dt,
                                               (nsteps+1)*dt, v1, v0,
                                               shifted_source)
                                         for x in range(reflector_x, nx)])

    model = tf.constant(model)
    sources = tf.constant(sources)

    sources_x = np.zeros([1, 1, 2], np.int32)
    sources_x[:, :, 1] = source_x
    sources_x = tf.constant(sources_x)

    out_wavefields = forward1d(model, sources, sources_x, dx, dt)

    sess = tf.Session()

    actual = sess.run(out_wavefields[-1, :])
    #return expected, actual.ravel()
    assert np.allclose(expected, actual.ravel(), atol=2.5)


def test_scatter_1d(dx=5, dt=0.0001, v0=1500, dv=50):
    """Create a constant model with one point scatterer, and the expected
    waveform at a point in a manner than depends nonlinearly on the
    scattering amplitude, and in an approximate linearised manner on it,
    and compare both with the forward propagated wave.
    """
    tf.reset_default_graph()
    nx = 100
    scatter_x = 80
    model_const = np.ones(nx, np.float32) * v0
    model_scatter = model_const.copy()
    model_scatter[scatter_x] = v0 + dv
    nt = int((3 * scatter_x * dx / v0 + 0.1) / dt)
    propagator = forward1d
    sources_x = np.ones([1, 1, 1], np.int)
    receivers_x = sources_x.copy()
    sources = ricker(25, nt, dt, 0.05).reshape([-1, 1, 1])
    actual = gen_data(model_scatter, dx, dt, sources, sources_x, receivers_x,
                      propagator)[1:-1]

    scatterer_x = np.ones([1, 1, 1], np.int)
    scatterer_x[0, 0, 0] = scatter_x
    d_sc = gen_data(model_const, dx, dt, sources, sources_x, scatterer_x,
                    propagator)
    d2d_scdt2 = (d_sc[:-2] - 2*d_sc[1:-1] + d_sc[2:])/dt**2
    sc_d_sc_nonlinear = -d2d_scdt2 * (v0**2 / (v0 + dv)**2 - 1) / v0**2
    sc_d_sc_linear = d2d_scdt2 * 2 * dv / v0**3

    expected_nonlinear = gen_data(model_const, dx, dt, sc_d_sc_nonlinear,
                                  scatterer_x, receivers_x, propagator)

    expected_linear = gen_data(model_const, dx, dt, sc_d_sc_linear,
                               scatterer_x, receivers_x, propagator)

    #return expected_nonlinear.ravel(), expected_linear.ravel(), actual.ravel()
    # The "expected" waveforms do not include the direct wave, so only
    # compare the later half of the waveforms (which should only include
    # the scattered arrival).
    st = nt//2
    assert np.allclose(expected_nonlinear.ravel()[st:], actual.ravel()[st:],
                       atol=0.04)
    assert np.allclose(expected_linear.ravel()[st:], actual.ravel()[st:],
                       atol=0.04)

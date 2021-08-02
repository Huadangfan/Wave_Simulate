"""1D scalar wave equation forward modeling implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf

class TimeStepCell(tf.contrib.rnn.RNNCell):
    """One forward modeling step of scalar wave equation with PML.

    Args:
        model_padded2_dt2: Tensor containing squared wave speed times squared
                           time step size
        dt: Float specifying time step size
        sigma: Tensor that is only non-zero in PML regions
        first_deriv: Function to calculate the first derivative of the input
                     1D Tensor
        second_deriv: Function to calculate the second derivative of the input
                      1D Tensor
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 2]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer x cell coordinate of
                   the source
    """

    def __init__(self, model_padded2_dt2, dt, sigma, first_deriv, second_deriv,
                 sources_x):
        super(TimeStepCell, self).__init__()
        self.model_padded2_dt2 = model_padded2_dt2
        self.dt = dt
        self.sigma = sigma
        self.first_deriv = first_deriv
        self.second_deriv = second_deriv
        self.sources_x = sources_x
        self.nx_padded = model_padded2_dt2.shape[0]

    @property
    def state_size(self):
        """The RNN state (passed between RNN units) contains two time steps
        of the wave field, and the PML auxiliary wavefield phi.
        """
        return [self.nx_padded, self.nx_padded, self.nx_padded]

    @property
    def output_size(self):
        """The output of the RNN unit contains one time step of the wavefield.
        """
        return self.nx_padded

    def __call__(self, inputs, state):
        """Propagate the wavefield forward one time step.

        Args:
            inputs: An array containing the source amplitudes for this time
                    step
            state: A list containing the two previous wave field time steps
                   and the auxiliary wavefield phi

        Returns:
            output: The current wave field
            state: A list containing the current and one previous wave field
                   time steps and the updated auxiliary wavefield phi
        """
        model_shape = tf.shape(state[0])
        wavefieldc = state[0]
        wavefieldp = state[1]
        phic = state[2]

        # The main evolution equation
        wavefieldf = (self.model_padded2_dt2 / (1 + self.dt * self.sigma/2)
                      * (self.second_deriv(wavefieldc) + self.first_deriv(phic))
                      + self.dt * self.sigma * wavefieldp
                      / (2 + self.dt * self.sigma)
                      + 1 / (1 + self.dt * self.sigma / 2)
                      * (2 * wavefieldc - wavefieldp))

        # Update PML variable phi
        phif = (-self.sigma * self.dt * self.first_deriv(wavefieldc) + phic
                - self.dt * self.sigma * phic)

        # Add the sources
        # f(t+1, x_s) += c(x_s)^2 * dt^2 * s(t)
        # We need to expand "inputs" to be the same size as f(t+1), so we
        # use tf.scatter_nd. This will create an array
        # of the right size, almost entirely filled with zeros, with the
        # source amplitudes (multiplied by c^2 * dt^2) in the right places.
        wavefieldf += tf.scatter_nd(self.sources_x, inputs, model_shape)

        return (tf.reshape(wavefieldf, model_shape),
                [tf.reshape(wavefieldf, model_shape),
                 tf.reshape(wavefieldc, model_shape),
                 tf.reshape(phif, model_shape)])


def forward1d(model, sources, sources_x,
              dx, dt, pml_width=None, pad_width=None,
              profile=None):
    """Forward modeling using the 1D wave equation.

    Args:
        model: 1D tf.Variable or tf.Tensor velocity model
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 2]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer x cell coordinate of
                   the source
        dx: float specifying size of each cell
        dt: float specifying time between time steps
        pml_width: number of cells in PML (optional)
        pad_width: number of padding cells outside PML (optional)
        profile: 1D array specifying PML profile (optional)

    Returns:
        3D Tensor [num_time_steps, batch_size, nx] containing time steps of
        wavefields. Padding that was added is removed.
    """

    if pml_width is None:
        pml_width = 10
    if pad_width is None:
        pad_width = 8

    total_pad = pml_width + pad_width

    nx_padded = _set_x(model, total_pad)

    model_padded2_dt2 = _set_model(model, total_pad, dt)

    profile, pml_width = _set_profile(profile, pml_width, dx)

    sigma = _set_sigma(nx_padded, total_pad, pad_width, profile)

    sources, sources_x = _set_sources(sources, sources_x, total_pad,
                                      model_padded2_dt2)

    d1_kernel, d2_kernel = _set_kernels(dx)

    first_deriv, second_deriv = _set_deriv_funcs(d1_kernel, d2_kernel)

    cell = TimeStepCell(model_padded2_dt2, dt, sigma,
                        first_deriv, second_deriv, sources_x)

    out, _ = tf.nn.dynamic_rnn(cell, sources,
                               dtype=tf.float32, time_major=True)

    return out[:, :, total_pad : -total_pad]


def _set_x(model, total_pad):
    """Calculate the size of the model after padding has been added.

    Args:
        model: 1D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge

    Returns:
        Integer specifying number of cells in padded model
    """
    nx = model.shape[0]
    nx_padded = nx + 2 * total_pad
    return nx_padded


def _set_model(model, total_pad, dt):
    """Add padding to the model (extending edge values) and compute c^2 * dt^2.

    Args:
        model: 1D tf.Variable or tf.Tensor velocity model
        total_pad: Integer specifying padding to add to each edge
        dt: Float specifying time step size

    Returns:
        A 1D Tensor containing the squared model times the squared time step
        size
    """
    model_padded = tf.pad(model, [[total_pad, 0]], 'CONSTANT',
                          constant_values=model[0])
    model_padded = tf.pad(model_padded, [[0, total_pad]], 'CONSTANT',
                          constant_values=model[-1])
    return tf.square(model_padded) * dt**2


def _set_profile(profile, pml_width, dx):
    """Create a profile for the PML.

    Args:
        profile: User supplied profile, if None use default
        pml_width: Integer. If profile is None, create a PML of this width.
        dx: Float specifying spacing between grid cells

    Returns:
        profile: 1D array containing PML profile
        pml_width: Integer specifying the length of the profile
    """
    # This should be set to approximately the maximum wave speed at the edges
    # of the model
    max_vel = 5000
    if profile is None:
        # See Collino & Tsogka, Geophysics (2001)
        profile = ((np.arange(pml_width)/pml_width)**2
                   * 3 * max_vel * np.log(1000)
                   / (2 * dx * pml_width))
    else:
        pml_width = len(profile)
    return profile, pml_width


def _set_sigma(nx_padded, total_pad, pad_width, profile):
    """Create a 1D sigma array that contains the PML profile in the PML regions

    Args:
        nx_padded: Integer specifying the number of cells in the padded model
        total_pad: Integer specifying the number of cells of padding added to
                   each edge of the model
        pad_width: Integer specifying the number of cells of padding that are
                   not part of the PML
        profile: 1D array containing the PML profile for the right side of
                 the model (for the left side, it will be reversed)

    Returns:
        1D sigma array
    """
    sigma = np.zeros(nx_padded, np.float32)
    sigma[total_pad-1:pad_width-1:-1] = profile
    sigma[-total_pad:-pad_width] = profile
    sigma[:pad_width] = sigma[pad_width]
    sigma[-pad_width:] = sigma[-pad_width-1]
    return tf.constant(sigma)


def _set_sources(sources, sources_x, total_pad, model_padded2_dt2):
    """Set the source amplitudes, and the source positions.

    Args:
        sources: 3D Tensor [num_time_steps, batch_size, num_sources_per_shot]
                 containing source amplitudes
        sources_x: 3D Tensor [batch_size, num_sources_per_shot, 2]
                   where [:, :, 0] contains the index in the batch, and
                   [:, :, 1] contains the integer x cell coordinate of
                   the source
        total_pad: Integer specifying padding added to each edge of the model
        model_padded2_dt2: Tensor containing squared wave speed times squared
                           time step size

    Returns:
        sources: 3D Tensor containing source amplitude * c^2 * dt^2
        sources_x: 3D Tensor like the input, but with total_pad added to
                   [:, :, 1]
    """
    # I add "total_pad" to the source coordinates as the coordinates currently
    # refer to the coordinates in the unpadded model, but we need them to
    # refer to the coordinates when padding has been added. We only want to add
    # this to [:, :, 1], which contains the x coordinates, so I multiply by
    # arange, which will be 0 for [:, :, 0] and 1 for [:, :, 1].
    sources_x += (tf.ones_like(sources_x) * total_pad
                  * np.arange(2).reshape([1, 1, 2]))

    # The propagator injected source amplitude multiplied by c(x)^2 * dt^2
    # at the locations of the sources, so we need to extract the wave speed
    # at these locations. I do this using tf.gather
    sources_v = tf.gather(model_padded2_dt2, sources_x[:, :, 1])

    # The propagator does not need the unmultiplied source amplitudes,
    # so I will save space by only storing the source amplitudes multiplied
    # by c(x)^2 * dt^2
    sources = sources * sources_v

    return sources, sources_x


def _set_kernels(dx):
    """Create spatial finite difference kernels.

    The kernels are reshaped into the appropriate shape for a 1D
    convolution, and saved as constant tensors.

    Args:
        dx: Float specifying the grid cell spacing

    Returns:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 1D second derivative
    """
    # First derivative
    d1_kernel = (np.array([1/12, -2/3, 0, 2/3, -1/12], np.float32)
                 / dx)
    d1_kernel = d1_kernel.reshape([-1, 1, 1])
    d1_kernel = tf.constant(d1_kernel)

    # Second derivative
    d2_kernel = (np.array([-1/12, 4/3, -5/2, 4/3, -1/12], np.float32)
                 / dx**2)
    d2_kernel = d2_kernel.reshape([-1, 1, 1])
    d2_kernel = tf.constant(d2_kernel)

    return d1_kernel, d2_kernel


def _set_deriv_funcs(d1_kernel, d2_kernel):
    """Create functions to apply first and second derivatives.

    Args:
        d1_kernel: 3D Tensor for 1D first derivative
        d2_kernel: 3D Tensor for 1D second derivative

    Returns:
        Functions for applying first and second derivatives
    """
    def make_deriv_func(kernel):
        """Returns a function that takes a derivative of its input."""
        def deriv(x):
            """Take a derivative of the input."""
            return tf.squeeze(tf.nn.conv1d(tf.expand_dims(x, -1),
                                           kernel, 1, 'SAME'))
        return deriv

    return make_deriv_func(d1_kernel), make_deriv_func(d2_kernel)

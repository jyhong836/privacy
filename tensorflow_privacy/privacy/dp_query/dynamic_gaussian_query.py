from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import gaussian_query


class ExpDynamicSumQuery(dp_query.SumAggregationDPQuery):
  """Dynamically adjust the noise stddev of Gaussian noise."""
  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', [
          'initial_noise_multiplier',
          'l2_norm_clip',
          'k',
          'step',
          'sum_state',  # state of gaussian sum query.
      ])

  # pylint: disable=invalid-name
  _SampleState = collections.namedtuple(
      '_SampleState', ['sum_state'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['sum_params'])

  def __init__(self, l2_norm_clip, initial_noise_multiplier, initial_step, k):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      initial_noise_multiplier: The initial noise multiplier.
      initial_step: The initial step for the dynamic function.
      k: The coefficient used in the dynamic function.
    """
    self._l2_norm_clip = l2_norm_clip
    self._initial_noise_multiplier = initial_noise_multiplier
    self._initial_step = initial_step
    self._k = k

    # Initialize sum query's global state with None, to be set later.
    self._sum_query = gaussian_query.GaussianSumQuery(None, None)

    self._ledger = None

  def set_ledger(self, ledger):
    """See base class."""
    self._sum_query.set_ledger(ledger)

  def initial_global_state(self):
    """See base class."""
    initial_noise_multiplier = tf.cast(self._initial_noise_multiplier, tf.float32)
    l2_norm_clip = tf.cast(self._l2_norm_clip, tf.float32)
    k = tf.cast(self._k, tf.float32)
    global_step = tf.cast(self._initial_step, tf.int32)
    sum_stddev = l2_norm_clip * initial_noise_multiplier

    sum_query_global_state = self._sum_query.make_global_state(
        l2_norm_clip=l2_norm_clip,
        stddev=sum_stddev)

    return self._GlobalState(
        initial_noise_multiplier,
        l2_norm_clip,
        k,
        global_step,
        sum_query_global_state)

  def derive_sample_params(self, global_state):
    """See base class."""
    # Assign values to variables that inner sum query uses.
    sum_params = self._sum_query.derive_sample_params(global_state.sum_state)
    return self._SampleParams(sum_params)

  def initial_sample_state(self, template):
    """See base class."""
    sum_state = self._sum_query.initial_sample_state(template)
    return self._SampleState(sum_state)

  def preprocess_record(self, params, record):
    preprocessed_sum_record, global_norm = (
        self._sum_query.preprocess_record_impl(params.sum_params, record))
    return self._SampleState(preprocessed_sum_record)

  def dynamic_noise_multiplier(self, initial_noise_multiplier, k, global_step):
    return initial_noise_multiplier * tf.exp(- k * tf.cast(global_step, tf.float32))

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    gs = global_state

    noised_vectors, sum_state = self._sum_query.get_noised_result(
        sample_state.sum_state, gs.sum_state)
    # del sum_state  # Unused. To be set explicitly later.

    new_noise_multiplier = self.dynamic_noise_multiplier(global_state.initial_noise_multiplier, global_state.k,
                                                         global_state.step)
    new_sum_stddev = global_state.l2_norm_clip * new_noise_multiplier
    new_sum_query_global_state = sum_state._replace(stddev=new_sum_stddev)

    new_global_state = global_state._replace(
        step=global_state.step + 1,
        sum_state=new_sum_query_global_state)

    return noised_vectors, new_global_state

# TODO add average query.

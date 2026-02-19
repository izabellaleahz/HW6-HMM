import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """Test forward + viterbi on the mini weather model, plus edge cases."""

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    hmm_model = HiddenMarkovModel(
        observation_states=mini_hmm['observation_states'],
        hidden_states=mini_hmm['hidden_states'],
        prior_p=mini_hmm['prior_p'],
        transition_p=mini_hmm['transition_p'],
        emission_p=mini_hmm['emission_p']
    )

    obs_seq = mini_input['observation_state_sequence']
    expected_hidden = list(mini_input['best_hidden_state_sequence'])

    # forward should give the right likelihood
    forward_prob = hmm_model.forward(obs_seq)
    assert np.isclose(forward_prob, 0.03506441162109375)

    # viterbi should recover the known best path
    viterbi_seq = hmm_model.viterbi(obs_seq)
    assert viterbi_seq == expected_hidden
    assert len(viterbi_seq) == len(obs_seq)

    # edge case: priors that don't sum to 1 should raise
    with pytest.raises(ValueError):
        HiddenMarkovModel(
            observation_states=mini_hmm['observation_states'],
            hidden_states=mini_hmm['hidden_states'],
            prior_p=np.array([0.5, 0.3]),  # sums to 0.8
            transition_p=mini_hmm['transition_p'],
            emission_p=mini_hmm['emission_p']
        )

    # edge case: transition matrix with wrong shape should raise
    with pytest.raises(ValueError):
        HiddenMarkovModel(
            observation_states=mini_hmm['observation_states'],
            hidden_states=mini_hmm['hidden_states'],
            prior_p=mini_hmm['prior_p'],
            transition_p=np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]),
            emission_p=mini_hmm['emission_p']
        )


def test_full_weather():
    """Test forward + viterbi on the full weather model."""

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    hmm_model = HiddenMarkovModel(
        observation_states=full_hmm['observation_states'],
        hidden_states=full_hmm['hidden_states'],
        prior_p=full_hmm['prior_p'],
        transition_p=full_hmm['transition_p'],
        emission_p=full_hmm['emission_p']
    )

    obs_seq = full_input['observation_state_sequence']
    expected_hidden = list(full_input['best_hidden_state_sequence'])

    viterbi_seq = hmm_model.viterbi(obs_seq)
    assert viterbi_seq == expected_hidden
    assert len(viterbi_seq) == len(obs_seq)

    # sanity check — forward prob should be positive
    forward_prob = hmm_model.forward(obs_seq)
    assert forward_prob > 0

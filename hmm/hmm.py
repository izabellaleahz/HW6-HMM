import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states
            hidden_states (np.ndarray): hidden states
            prior_p (np.ndarray): prior probabities of hidden states
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states
        """

        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}

        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        # check that dimensions all agree
        N = len(hidden_states)
        M = len(observation_states)

        if len(prior_p) != N:
            raise ValueError(f"prior_p length {len(prior_p)} doesn't match hidden states ({N})")
        if transition_p.shape != (N, N):
            raise ValueError(f"transition_p shape {transition_p.shape} should be ({N}, {N})")
        if emission_p.shape != (N, M):
            raise ValueError(f"emission_p shape {emission_p.shape} should be ({N}, {M})")

        # make sure probabilities are valid
        if not np.isclose(np.sum(prior_p), 1.0):
            raise ValueError(f"prior_p sums to {np.sum(prior_p)}, not 1.0")
        for i in range(N):
            if not np.isclose(np.sum(transition_p[i]), 1.0):
                raise ValueError(f"transition_p row {i} sums to {np.sum(transition_p[i])}")
            if not np.isclose(np.sum(emission_p[i]), 1.0):
                raise ValueError(f"emission_p row {i} sums to {np.sum(emission_p[i])}")


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence
        """

        if len(input_observation_states) == 0:
            return 0.0

        T = len(input_observation_states)
        N = len(self.hidden_states)

        # alpha[t][s] = P(obs_0...obs_t, state_t = s)
        alpha = np.zeros((T, N))

        # base case: prior * emission for first observation
        obs_0 = self.observation_states_dict[input_observation_states[0]]
        for s in range(N):
            alpha[0][s] = self.prior_p[s] * self.emission_p[s][obs_0]

        # recursion
        for t in range(1, T):
            obs_t = self.observation_states_dict[input_observation_states[t]]
            for s in range(N):
                alpha[t][s] = sum(alpha[t-1][prev] * self.transition_p[prev][s] for prev in range(N)) * self.emission_p[s][obs_t]

        # total probability is sum over final states
        return float(np.sum(alpha[-1]))
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        if len(decode_observation_states) == 0:
            return []

        T = len(decode_observation_states)
        N = len(self.hidden_states)

        viterbi_table = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)

        # base case
        obs_0 = self.observation_states_dict[decode_observation_states[0]]
        for s in range(N):
            viterbi_table[0][s] = self.prior_p[s] * self.emission_p[s][obs_0]
            backpointer[0][s] = 0

        # recursion — same as forward but take max instead of sum
        for t in range(1, T):
            obs_t = self.observation_states_dict[decode_observation_states[t]]
            for s in range(N):
                max_prob = -1
                max_prev = 0
                for prev in range(N):
                    prob = viterbi_table[t-1][prev] * self.transition_p[prev][s]
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev
                viterbi_table[t][s] = max_prob * self.emission_p[s][obs_t]
                backpointer[t][s] = max_prev

        # traceback from best final state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_table[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1][best_path[t+1]]

        # convert indices back to state names
        best_hidden_state_sequence = [self.hidden_states_dict[idx] for idx in best_path]
        return best_hidden_state_sequence
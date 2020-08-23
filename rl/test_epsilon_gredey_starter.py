import pytest
import epsilon_greedy_starter
import statistics


@pytest.mark.parametrize("test_input", [([1, 2, 3, 4]), ([1])])
def test_compute_new_mean(test_input):
    expected_output = statistics.mean(test_input)

    previous_mean = statistics.mean(test_input[:-1]) if test_input[:-1] else 0
    newest_sample = test_input[-1]
    total_samples = len(test_input)

    test_ouput = epsilon_greedy_starter._compute_new_mean(
        previous_mean, newest_sample, total_samples
    )

    assert test_ouput == expected_output

import deep_lib.math as dlm
import numpy as np
import numpy.testing as npt


class TestMath(object):

    def test_sigmoid_function_with_cache(self):
        x = np.array([0])
        assert dlm.sigmoid(x)[0] == 0.5
        assert dlm.sigmoid(x)[1] == 0

        x = np.array([1])
        assert dlm.sigmoid(x)[0] > 0.5
        assert dlm.sigmoid(x)[1] == 1

        x = np.array([-1])
        assert dlm.sigmoid(x)[0] < 0.5
        assert dlm.sigmoid(x)[1] == -1

        x = np.array([-1, 0, 1])
        npt.assert_allclose(dlm.sigmoid(x)[0], [0.26, 0.5, 0.73], atol=0.01)
        npt.assert_allclose(dlm.sigmoid(x)[1], [-1, 0, 1])

from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import logsumexp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ops'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibrate_validation import temporal_folds,fit_temperature,calibration_report


class CalibrationTests(unittest.TestCase):
    def test_chronological_folds_keep_ties_together(self):
        times = np.array([30,10,20,20,40,50])
        fit,audit,boundary = temporal_folds(times)
        self.assertLess(times[fit].max(),times[audit].min())
        self.assertEqual(boundary,30)
        self.assertFalse(set(fit)&set(audit))
        self.assertEqual(len(fit)+len(audit),len(times))
        with self.assertRaises(ValueError): temporal_folds([1,1,1,1])

    def test_temperature_fit_and_uniform_ties(self):
        # 80% correct with overconfident +/-4 log odds: optimum = 4/log(4).
        scores = np.array([[4.,0.]]*80+[[0.,4.]]*20)
        temperature = fit_temperature(scores)
        self.assertAlmostEqual(temperature,4/np.log(4),places=4)
        nll = lambda x: (logsumexp(x,axis=1)-x[:,0]).mean()
        self.assertLess(nll(scores/temperature),nll(scores))
        report = calibration_report(np.zeros((4,100)),1.,np.arange(4))['overall']
        self.assertAlmostEqual(report['ece'],0.,places=12)
        self.assertAlmostEqual(report['nll'],np.log(100))
        self.assertAlmostEqual(report['brier_multiclass'],.99)
        with self.assertRaises(ValueError): fit_temperature([[np.nan,0.]])


if __name__=='__main__': unittest.main()

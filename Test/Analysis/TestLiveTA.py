import unittest
from App.Helpers.LiveTA import LiveTA
from App.Library.Analysis.MomentumAnalyses import MomentumAnalyses
from App.Library.Connection.Client import Client


class TestLiveTA(unittest.TestCase):

    def test_liveTA(self):
        # Create a new LiveTA object, this will retrieve the candles and run the TA library. Do this ONCE every cycle
        liveTA = LiveTA(window_size=240)

        # You can now use the LivaTA object to retrieve the required columns:
        momentum = MomentumAnalyses(liveTA)
        rsi = momentum.get_RSI(offset=None, window_size=None)  # Offset and window are set during LiveTA initialization

    @classmethod
    def tearDownClass(cls):
        Client.logout()

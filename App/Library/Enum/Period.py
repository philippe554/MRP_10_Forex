from enum import Enum


class Period:
    """
        Period constants for the fxcm library
        https://www.fxcm.com/fxcmpy/02_historical_data.html#Data-Frequency
    """

    MINUTE_1 = 'm1',
    MINUTE_5 = 'm5',
    MINUTE_15 = 'm15',
    MINUTE_30 = 'm30',

    HOUR_1 = 'H1',
    HOUR_2 = 'H2',
    HOUR_3 = 'H3',
    HOUR_4 = 'H4',
    HOUR_6 = 'H6',
    HOUR_8 = 'H8',

    DAY_1 = 'D1',

    WEEK_1 = 'W1',

    MONTH_1 = 'M1',


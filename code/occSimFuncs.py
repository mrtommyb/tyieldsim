from __future__ import division, print_function
import numpy as np
# import matplotlib.pyplot as plt
from numpy import random
# from numpy import ma
from scipy.interpolate import interp1d

msun = 1.9891E30
rsun = 695500000.
G = 6.67384E-11
AU = 149597870700.


def per2ars(per, mstar, rstar):
    per_SI = per * 86400.
    mass_SI = mstar * msun
    a3 = per_SI**2 * G * mass_SI / (4 * np.pi**2)
    return a3**(1. / 3.) / (rstar * rsun)


def get_duration(per, ars, cosi=0.0, b=0, rprs=0.0):
    """
    returns the transit duration in days
    """
    part1 = (per / np.pi)
    part2 = 1. / ars
    part3 = np.sqrt((1 + rprs)**2 - b**2)
    part4 = np.sqrt(1 - cosi**2)
    duration = part1 * np.arcsin(part2 * part3 / part4)

    return duration


def get_transit_depth(Prad, rstar_solar):
    """
    returns transit depth in ppm
    """
    tdep = (Prad*  0.009155 / rstar_solar)**2 * 1.E6 #ppm
    return tdep


def get_rprs(Prad, rstar_solar):
    return (Prad * 0.009155) / rstar_solar


def Dressing_select(nselect=1):
    # create a pot for dressing numbers (balls)
    balls = np.array([])
    # there are 15 pots total of 1000 balls
    # pot zero means no planet
    p0 = np.zeros(91)
    balls = np.r_[balls, p0]
    # pot 1 contains rp=0.5-0.7, p=0.68-10
    p1 = np.zeros(14) + 1
    # pot 2 contains rp=0.7-1.0, p=0.68-10
    p2 = np.zeros(109) + 2
    # pot 3 contains rp=1.0-1.4, p=0.68-10
    p3 = np.zeros(108) + 3
    #pot 4 contains rp=1.4-2.0, p=0.68-10
    p4 = np.zeros(80) + 4
    #pot 5 contains rp=2.0-2.8, p=0.68-10
    p5 = np.zeros(38) + 5
    #pot 6 contains rp=2.8-4.0, p=0.68-10
    p6 = np.zeros(5) + 6
    #pot 7 contains rp=4.0-5.7, p=0.68-10
    p7 = np.zeros(4) + 7
    #pot 8 contains rp=0.5-0.7, p=10-50  = none
    #pot 9 contains rp=0.7-1.0, p=10-50
    p9 = np.zeros(103) + 9
    #pot 10 contains rp=1.0-1.4, p=10-50
    p10 = np.zeros(177) + 10
    #pot 11 contains rp=1.4-2.0, p=10-50
    p11 = np.zeros(123) + 11
    #pot 12 contains rp=2.0-2.8, p=10-50
    p12 = np.zeros(148) + 12

    balls = np.r_[balls,p1,p2,p3,p4,p5,p6,p7,p9,
                 p10,p11,p12]

    #lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    ball_lookup = {
                    0: [0.0,0.0,0.0,0.0],
                    1: [0.5,0.7,0.68,10.],
                    2: [0.7,1.0,0.68,10.],
                    3: [1.0,1.4,0.68,10.],
                    4: [1.4,2.0,0.68,10.],
                    5: [2.0,2.8,0.68,10.],
                    6: [2.8,4.0,0.68,10.],
                    7: [4.0,5.7,0.68,10.],
                    9: [0.7,1.0,10.,50.],
                    10: [1.0,1.4,10.,50.],
                    11: [1.4,2.0,10.,50.],
                    12: [2.0,2.8,10.,50.]
                   }

    rsamps = random.choice(balls,size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i,samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]
        radius[i] = random.uniform(low=rl,high=rh)
        period[i] = random.uniform(low=pl,high=ph)

    return radius, period

def Dressing15_select(nselect=1):
    """

    period bins = 0.5, 0.91, 1.66, 3.02, 5.49, 10.0, 18.2, 33.1, 60.3, 110., 200.
    """
    #create a pot for dressing numbers (balls)
    balls = np.array([])
    #pot 1 contains rp=0.5-1.0, p=0.5-0.91
    p1 = np.zeros(400) + 1
    #pot 2 contains rp=1.0-1.5, p=0.5-0.91
    p2 = np.zeros(460) + 2
    #pot 3 contains rp=1.5-2.0, p=0.5-0.91
    p3 = np.zeros(61) + 3
    #pot 4 contains rp=2.0-2.5, p=0.5-0.91
    p4 = np.zeros(2) + 4
    #pot 5 contains rp=2.5-3.0, p=0.5-0.91
    p5 = np.zeros(0) + 5
    #pot 6 contains rp=3.0-3.5, p=0.5-0.91
    p6 = np.zeros(0) + 6
    #pot 7 contains rp=3.5-4.0, p=0.5-0.91
    p7 = np.zeros(0) + 7
    #pot 1 contains rp=0.5-1.0, p=0.91, 1.66
    p8 = np.zeros(1500) + 8
    #pot 2 contains rp=1.0-1.5, p=0.91, 1.66
    p9 = np.zeros(1400) + 9
    #pot 3 contains rp=1.5-2.0, p=0.91, 1.66
    p10 = np.zeros(270) + 10
    #pot 4 contains rp=2.0-2.5, p=0.91, 1.66
    p11 = np.zeros(9) + 11
    #pot 5 contains rp=2.5-3.0, p=0.91, 1.66
    p12 = np.zeros(4) + 12
    #pot 6 contains rp=3.0-3.5, p=0.91, 1.66
    p13 = np.zeros(6) + 13
    #pot 7 contains rp=3.5-4.0, p=0.91, 1.66
    p14 = np.zeros(8) + 14
    #pot 1 contains rp=0.5-1.0, p=1.66, 3.02
    p15 = np.zeros(4400) + 15
    #pot 2 contains rp=1.0-1.5, p=1.66, 3.02
    p16 = np.zeros(3500) + 16
    #pot 3 contains rp=1.5-2.0, p=1.66, 3.02
    p17 = np.zeros(1200) + 17
    #pot 4 contains rp=2.0-2.5, p=1.66, 3.02
    p18 = np.zeros(420) + 18
    #pot 5 contains rp=2.5-3.0, p=1.66, 3.02
    p19 = np.zeros(230) + 19
    #pot 6 contains rp=3.0-3.5, p=1.66, 3.02
    p20 = np.zeros(170) + 20
    #pot 7 contains rp=3.5-4.0, p=1.66, 3.02
    p21 = np.zeros(180) + 21
    #pot 1 contains rp=0.5-1.0, p=3.02, 5.49
    p22 = np.zeros(5500) + 22
    #pot 2 contains rp=1.0-1.5, p=3.02, 5.49
    p23 = np.zeros(5700) + 23
    #pot 3 contains rp=1.5-2.0, p=3.02, 5.49
    p24 = np.zeros(2500) + 24
    #pot 4 contains rp=2.0-2.5, p=3.02, 5.49
    p25 = np.zeros(1800) + 25
    #pot 5 contains rp=2.5-3.0, p=3.02, 5.49
    p26 = np.zeros(960) + 26
    #pot 6 contains rp=3.0-3.5, p=3.02, 5.49
    p27 = np.zeros(420) + 27
    #pot 7 contains rp=3.5-4.0, p=3.02, 5.49
    p28 = np.zeros(180) + 28
    #pot 1 contains rp=0.5-1.0, p=5.49, 10.0
    p29 = np.zeros(10000) + 29
    #pot 2 contains rp=1.0-1.5, p=5.49, 10.0
    p30 = np.zeros(10000) + 30
    #pot 3 contains rp=1.5-2.0, p=5.49, 10.0
    p31 = np.zeros(6700) + 31
    #pot 4 contains rp=2.0-2.5, p=5.49, 10.0
    p32 = np.zeros(6400) + 32
    #pot 5 contains rp=2.5-3.0, p=5.49, 10.0
    p33 = np.zeros(2700) + 33
    #pot 6 contains rp=3.0-3.5, p=5.49, 10.0
    p34 = np.zeros(1100) + 34
    #pot 7 contains rp=3.5-4.0, p=5.49, 10.0
    p35 = np.zeros(360) + 35
    #pot 1 contains rp=0.5-1.0, p=10.0, 18.2
    p36 = np.zeros(12000) + 36
    #pot 2 contains rp=1.0-1.5, p=10.0, 18.2
    p37 = np.zeros(13000) + 37
    #pot 3 contains rp=1.5-2.0, p=10.0, 18.2
    p38 = np.zeros(13000) + 38
    #pot 4 contains rp=2.0-2.5, p=10.0, 18.2
    p39 = np.zeros(9300) + 39
    #pot 5 contains rp=2.5-3.0, p=10.0, 18.2
    p40 = np.zeros(3800) + 40
    #pot 6 contains rp=3.0-3.5, p=10.0, 18.2
    p41 = np.zeros(1400) + 41
    #pot 7 contains rp=3.5-4.0, p=10.0, 18.2
    p42 = np.zeros(510) + 42
    #pot 1 contains rp=0.5-1.0, p=18.2, 33.1
    p43 = np.zeros(11000) + 43
    #pot 2 contains rp=1.0-1.5, p=18.2, 33.1
    p44 = np.zeros(16000) + 44
    #pot 3 contains rp=1.5-2.0, p=18.2, 33.1
    p45 = np.zeros(14000) + 45
    #pot 4 contains rp=2.0-2.5, p=18.2, 33.1
    p46 = np.zeros(10000) + 46
    #pot 5 contains rp=2.5-3.0, p=18.2, 33.1
    p47 = np.zeros(4600) + 47
    #pot 6 contains rp=3.0-3.5, p=18.2, 33.1
    p48 = np.zeros(810) + 48
    #pot 7 contains rp=3.5-4.0, p=18.2, 33.1
    p49 = np.zeros(320) + 49
    #pot 1 contains rp=0.5-1.0, p=33.1, 60.3
    p50 = np.zeros(6400) + 50
    #pot 2 contains rp=1.0-1.5, p=33.1, 60.3
    p51 = np.zeros(6400) + 51
    #pot 3 contains rp=1.5-2.0, p=33.1, 60.3
    p52 = np.zeros(12000) + 52
    #pot 4 contains rp=2.0-2.5, p=33.1, 60.3
    p53 = np.zeros(12000) + 53
    #pot 5 contains rp=2.5-3.0, p=33.1, 60.3
    p54 = np.zeros(5800) + 54
    #pot 6 contains rp=3.0-3.5, p=33.1, 60.3
    p55 = np.zeros(1600) + 55
    #pot 7 contains rp=3.5-4.0, p=33.1, 60.3
    p56 = np.zeros(210) + 56
    #pot 1 contains rp=0.5-1.0, p=60.3, 110.
    p57 = np.zeros(10000) + 57
    #pot 2 contains rp=1.0-1.5, p=60.3, 110.
    p58 = np.zeros(10000) + 58
    #pot 3 contains rp=1.5-2.0, p=60.3, 110.
    p59 = np.zeros(8300) + 59
    #pot 4 contains rp=2.0-2.5, p=60.3, 110.
    p60 = np.zeros(9600) + 60
    #pot 5 contains rp=2.5-3.0, p=60.3, 110.
    p61 = np.zeros(4200) + 61
    #pot 6 contains rp=3.0-3.5, p=60.3, 110.
    p62 = np.zeros(1700) + 62
    #pot 7 contains rp=3.5-4.0, p=60.3, 110.
    p63 = np.zeros(420) + 63
    #pot 1 contains rp=0.5-1.0, p=110., 200.
    p64 = np.zeros(19000) + 64
    #pot 2 contains rp=1.0-1.5, p=110., 200.
    p65 = np.zeros(19000) + 65
    #pot 3 contains rp=1.5-2.0, p=110., 200.
    p66 = np.zeros(10000) + 66
    #pot 4 contains rp=2.0-2.5, p=110., 200.
    p67 = np.zeros(4500) + 67
    #pot 5 contains rp=2.5-3.0, p=110., 200.
    p68 = np.zeros(1100) + 68
    #pot 6 contains rp=3.0-3.5, p=110., 200.
    p69 = np.zeros(160) + 69
    #pot 7 contains rp=3.5-4.0, p=110., 200.
    p70 = np.zeros(80) + 70
    

    balls = np.r_[balls,
                    p1,p2,p3,p4,p5,p6,p7,p9,p10,
                    p11,p12,p13,p14,p15,p16,p17,p19,p20,
                    p21,p22,p23,p24,p25,p26,p27,p29,p30,
                    p31,p32,p33,p34,p35,p36,p37,p39,p40,
                    p41,p42,p43,p44,p45,p46,p47,p49,p50,
                    p51,p52,p53,p54,p55,p56,p57,p59,p60,
                    p61,p62,p63,p64,p65,p66,p67,p69,p70,
                    ]

    #lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    # 0.5, 0.91, 1.66, 3.02, 5.49, 10.0, 18.2, 33.1, 60.3, 110., 200.
    ball_lookup = {
                    1:  [0.5,1.0,0.5, 0.91],
                    2:  [1.0,1.5,0.5, 0.91],
                    3:  [1.5,2.0,0.5, 0.91],
                    4:  [2.0,2.5,0.5, 0.91],
                    5:  [2.5,3.0,0.5, 0.91],
                    6:  [3.0,3.5,0.5, 0.91],
                    7:  [3.5,4.0,0.5, 0.91],
                    8:  [0.5,1.0,0.91, 1.66],
                    9:  [1.0,1.5,0.91, 1.66],
                    10: [1.5,2.0,0.91, 1.66],
                    11: [2.0,2.5,0.91, 1.66],
                    12: [2.5,3.0,0.91, 1.66],
                    13: [3.0,3.5,0.91, 1.66],
                    14: [3.5,4.0,0.91, 1.66],
                    15: [0.5,1.0,1.66, 3.02],
                    16: [1.0,1.5,1.66, 3.02],
                    17: [1.5,2.0,1.66, 3.02],
                    18: [2.0,2.5,1.66, 3.02],
                    19: [2.5,3.0,1.66, 3.02],
                    20: [3.0,3.5,1.66, 3.02],
                    21: [3.5,4.0,1.66, 3.02],
                    22: [0.5,1.0,3.02, 5.49],
                    23: [1.0,1.5,3.02, 5.49],
                    24: [1.5,2.0,3.02, 5.49],
                    25: [2.0,2.5,3.02, 5.49],
                    26: [2.5,3.0,3.02, 5.49],
                    27: [3.0,3.5,3.02, 5.49],
                    28: [3.5,4.0,3.02, 5.49],
                    29: [0.5,1.0,5.49, 10.0],
                    30: [1.0,1.5,5.49, 10.0],
                    31: [1.5,2.0,5.49, 10.0],
                    32: [2.0,2.5,5.49, 10.0],
                    33: [2.5,3.0,5.49, 10.0],
                    34: [3.0,3.5,5.49, 10.0],
                    35: [3.5,4.0,5.49, 10.0],
                    36: [0.5,1.0,10.0, 18.2],
                    37: [1.0,1.5,10.0, 18.2],
                    38: [1.5,2.0,10.0, 18.2],
                    39: [2.0,2.5,10.0, 18.2],
                    40: [2.5,3.0,10.0, 18.2],
                    41: [3.0,3.5,10.0, 18.2],
                    42: [3.5,4.0,10.0, 18.2],
                    43: [0.5,1.0,18.2, 33.1],
                    44: [1.0,1.5,18.2, 33.1],
                    45: [1.5,2.0,18.2, 33.1],
                    46: [2.0,2.5,18.2, 33.1],
                    47: [2.5,3.0,18.2, 33.1],
                    48: [3.0,3.5,18.2, 33.1],
                    49: [3.5,4.0,18.2, 33.1],
                    50: [0.5,1.0,33.1, 60.3],
                    51: [1.0,1.5,33.1, 60.3],
                    52: [1.5,2.0,33.1, 60.3],
                    53: [2.0,2.5,33.1, 60.3],
                    54: [2.5,3.0,33.1, 60.3],
                    55: [3.0,3.5,33.1, 60.3],
                    56: [3.5,4.0,33.1, 60.3],
                    57: [0.5,1.0,60.3, 110.],
                    58: [1.0,1.5,60.3, 110.],
                    59: [1.5,2.0,60.3, 110.],
                    60: [2.0,2.5,60.3, 110.],
                    61: [2.5,3.0,60.3, 110.],
                    62: [3.0,3.5,60.3, 110.],
                    63: [3.5,4.0,60.3, 110.],
                    64: [0.5,1.0,110., 200.],
                    65: [1.0,1.5,110., 200.],
                    66: [1.5,2.0,110., 200.],
                    67: [2.0,2.5,110., 200.],
                    68: [2.5,3.0,110., 200.],
                    69: [3.0,3.5,110., 200.],
                    70: [3.5,4.0,110., 200.],
                   }

    rsamps = random.choice(balls,size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i,samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]
        radius[i] = random.uniform(low=rl,high=rh)
        period[i] = random.uniform(low=pl,high=ph)

    return radius, period

def Fressin_select(nselect=1):
    #create a pot for dressing numbers (balls)
    balls = np.array([])
    # there are 36 pots total of 1000 balls
    # pot zero means no planet
    p0 = np.zeros(447)
    balls = np.r_[balls,p0]
    #pot 1 contains rp=0.8-0.1.25, p=0.8-2
    p1 = np.zeros(2) + 1
    #pot 2 contains rp=1.25-2.0, p=0.8-2
    p2 = np.zeros(2) + 2
    #pot 3 contains rp=2-4, p=0.8-2
    p3 = np.zeros(0) + 3
    #pot 4 contains rp=4-6, p=0.8-2
    p4 = np.zeros(0) + 4
    #pot 5 contains rp=6-22, p=0.8-2
    p5 = np.zeros(0) + 5

    #pot 6 contains rp=0.8-0.1.25, p=2-3.4
    p6 = np.zeros(6) + 6
    #pot 7 contains rp=1.25-2.0, p=2-3.4
    p7 = np.zeros(7) + 7
    #pot 8 contains rp=2-4, p=2-3.4
    p8 = np.zeros(2) + 8
    #pot 9 contains rp=4-6, p=2-3.4
    p9 = np.zeros(0) + 9
    #pot 10 contains rp=6-22, p=2-3.4
    p10 = np.zeros(1) + 10

    #pot 11 contains rp=0.8-0.1.25, p=3.4-5.9
    p11 = np.zeros(17) + 11
    #pot 12 contains rp=1.25-2.0, p=3.4-5.9
    p12 = np.zeros(15) + 12
    #pot 13 contains rp=2-4, p=3.4-5.9
    p13 = np.zeros(7) + 13
    #pot 14 contains rp=4-6, p=3.4-5.9
    p14 = np.zeros(1) + 14
    #pot 15 contains rp=6-22, p=3.4-5.9
    p15 = np.zeros(2) + 15

    #pot 16 contains rp=0.8-0.1.25, p=5.9-10
    p16 = np.zeros(27) + 16
    #pot 17 contains rp=1.25-2.0, p=5.9-10
    p17 = np.zeros(29) + 17
    #pot 18 contains rp=2-4, p=5.9-10
    p18 = np.zeros(19) + 18
    #pot 19 contains rp=4-6, p=5.9-10
    p19 = np.zeros(1) + 19
    #pot 20 contains rp=6-22, p=5.9-10
    p20 = np.zeros(2) + 20

    #pot 21 contains rp=0.8-0.1.25, p=10-17
    p21 = np.zeros(27) + 21
    #pot 22 contains rp=1.25-2.0, p=10-17
    p22 = np.zeros(43) + 22
    #pot 23 contains rp=2-4, p=10-17
    p23 = np.zeros(37) + 23
    #pot 24 contains rp=4-6, p=10-17
    p24 = np.zeros(3) + 24
    #pot 25 contains rp=6-22, p=10-17
    p25 = np.zeros(3) + 25

    #pot 26 contains rp=0.8-0.1.25, p=17-29
    p26 = np.zeros(29) + 26
    #pot 27 contains rp=1.25-2.0, p=17-29
    p27 = np.zeros(45) + 27
    #pot 28 contains rp=2-4, p=17-29
    p28 = np.zeros(53) + 28
    #pot 29 contains rp=4-6, p=17-29
    p29 = np.zeros(3) + 29
    #pot 30 contains rp=6-22, p=17-29
    p30 = np.zeros(2) + 30

    #pot 31 contains rp=0.8-0.1.25, p=29-59
    p31 = np.zeros(41) + 31
    #pot 32 contains rp=1.25-2.0, p=29-59
    p32 = np.zeros(53) + 32
    #pot 33 contains rp=2-4, p=29-59
    p33 = np.zeros(65) + 33
    #pot 34 contains rp=4-6, p=29-59
    p34 = np.zeros(5) + 34
    #pot 35 contains rp=6-22, p=29-59
    p35 = np.zeros(4) + 35

    balls = np.r_[balls,p1,p2,p3,p4,p5,p6,p7,p8,p9,
                 p10,p11,p12,p13,p14,p15,p16,p17,
                 p18,p19,p20,p21,p22,p23,p24,p25,
                 p26,p27,p28,p29,p30,p31,p32,p33,
                 p34,p35]
    assert(len(balls == 1000))

    #lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    ball_lookup = {
                    0: [0.0,0.0,0.0,0.0],
                    1: [0.8,1.25,0.8,2.],
                    2: [1.25,2.0,0.8,2.],
                    3: [2.0,4.0,0.8,2.],
                    4: [4.0,6.0,0.8,2.],
                    5: [6.0,22.,0.8,2.], #giant planets
                    6: [0.8,1.25,2.,3.4],
                    7: [1.25,2.0,2.,3.4],
                    8: [2.0,4.0,2.,3.4],
                    9: [4.0,6.0,2.,3.4],
                    10: [6.0,22.,2.,3.4], #giant planets
                    11: [0.8,1.25,3.4,5.9],
                    12: [1.25,2.0,3.4,5.9],
                    13: [2.0,4.0,3.4,5.9],
                    14: [4.0,6.0,3.4,5.9],
                    15: [6.0,22.,3.4,5.9], #giant planets
                    16: [0.8,1.25,5.9,10.],
                    17: [1.25,2.0,5.9,10.],
                    18: [2.0,4.0,5.9,10.],
                    19: [4.0,6.0,5.9,10.],
                    20: [6.0,22.,5.9,10.], #giant planets
                    21: [0.8,1.25,10.,17.],
                    22: [1.25,2.0,10.,17.],
                    23: [2.0,4.0,10.,17.],
                    24: [4.0,6.0,10.,17.],
                    25: [6.0,22.,10.,17.], #giant planets
                    26: [0.8,1.25,17.,29.],
                    27: [1.25,2.0,17.,29.],
                    28: [2.0,4.0,17.,29.],
                    29: [4.0,6.0,17.,29.],
                    30: [6.0,22.,17.,29.], #giant planets
                    31: [0.8,1.25,29.,50.],
                    32: [1.25,2.0,29.,50.],
                    33: [2.0,4.0,29.,50.],
                    34: [4.0,6.0,29.,50.],
                    35: [6.0,22.,29.,50.], #giant planets
                   }

    rsamps = random.choice(balls,size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i,samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]

        if samp in [5,10,15,20,25,30,35]:
            #check for giant planets
            # if a giant planet than draw from  a uniform in log
            radius[i] = np.e**random.uniform(low=np.log(rl),high=np.log(rh))
        else:
            radius[i] = random.uniform(low=rl,high=rh)
        
        period[i] = random.uniform(low=pl,high=ph)

    return radius, period

def Fressin13_select(nselect=1):
    #create a pot for dressing numbers (balls)
    balls = np.array([])
    #pot 1 contains rp=0.8-0.1.25, p=0.8-2
    p1 = np.zeros(180) + 1
    #pot 2 contains rp=1.25-2.0, p=0.8-2
    p2 = np.zeros(170) + 2
    #pot 3 contains rp=2-4, p=0.8-2
    p3 = np.zeros(35) + 3
    #pot 4 contains rp=4-6, p=0.8-2
    p4 = np.zeros(4) + 4
    #pot 5 contains rp=6-22, p=0.8-2
    p5 = np.zeros(15) + 5

    #pot 6 contains rp=0.8-0.1.25, p=2-3.4
    p6 = np.zeros(610) + 6
    #pot 7 contains rp=1.25-2.0, p=2-3.4
    p7 = np.zeros(740) + 7
    #pot 8 contains rp=2-4, p=2-3.4
    p8 = np.zeros(180) + 8
    #pot 9 contains rp=4-6, p=2-3.4
    p9 = np.zeros(6) + 9
    #pot 10 contains rp=6-22, p=2-3.4
    p10 = np.zeros(67) + 10

    #pot 11 contains rp=0.8-0.1.25, p=3.4-5.9
    p11 = np.zeros(1720) + 11
    #pot 12 contains rp=1.25-2.0, p=3.4-5.9
    p12 = np.zeros(1490) + 12
    #pot 13 contains rp=2-4, p=3.4-5.9
    p13 = np.zeros(730) + 13
    #pot 14 contains rp=4-6, p=3.4-5.9
    p14 = np.zeros(110) + 14
    #pot 15 contains rp=6-22, p=3.4-5.9
    p15 = np.zeros(170) + 15

    #pot 16 contains rp=0.8-0.1.25, p=5.9-10
    p16 = np.zeros(2700) + 16
    #pot 17 contains rp=1.25-2.0, p=5.9-10
    p17 = np.zeros(2900) + 17
    #pot 18 contains rp=2-4, p=5.9-10
    p18 = np.zeros(1930) + 18
    #pot 19 contains rp=4-6, p=5.9-10
    p19 = np.zeros(91) + 19
    #pot 20 contains rp=6-22, p=5.9-10
    p20 = np.zeros(180) + 20

    #pot 21 contains rp=0.8-0.1.25, p=10-17
    p21 = np.zeros(2700) + 21
    #pot 22 contains rp=1.25-2.0, p=10-17
    p22 = np.zeros(4300) + 22
    #pot 23 contains rp=2-4, p=10-17
    p23 = np.zeros(3670) + 23
    #pot 24 contains rp=4-6, p=10-17
    p24 = np.zeros(290) + 24
    #pot 25 contains rp=6-22, p=10-17
    p25 = np.zeros(270) + 25

    #pot 26 contains rp=0.8-0.1.25, p=17-29
    p26 = np.zeros(2930) + 26
    #pot 27 contains rp=1.25-2.0, p=17-29
    p27 = np.zeros(4490) + 27
    #pot 28 contains rp=2-4, p=17-29
    p28 = np.zeros(5290) + 28
    #pot 29 contains rp=4-6, p=17-29
    p29 = np.zeros(320) + 29
    #pot 30 contains rp=6-22, p=17-29
    p30 = np.zeros(230) + 30

    #pot 31 contains rp=0.8-0.1.25, p=29-50
    p31 = np.zeros(4080) + 31
    #pot 32 contains rp=1.25-2.0, p=29-50
    p32 = np.zeros(5290) + 32
    #pot 33 contains rp=2-4, p=29-50
    p33 = np.zeros(6450) + 33
    #pot 34 contains rp=4-6, p=29-50
    p34 = np.zeros(490) + 34
    #pot 35 contains rp=6-22, p=29-50
    p35 = np.zeros(350) + 35

    #pot 36 contains rp=0.8-0.1.25, p=50-85
    p36 = np.zeros(3460) + 36
    #pot 37 contains rp=1.25-2.0, p=50-85
    p37 = np.zeros(3660) + 37
    #pot 38 contains rp=2-4, p=50-85
    p38 = np.zeros(5250) + 38
    #pot 39 contains rp=4-6, p=50-85
    p39 = np.zeros(660) + 39
    #pot 40 contains rp=6-22, p=50-85
    p40 = np.zeros(710) + 40

    balls = np.r_[balls,p1,p2,p3,p4,p5,p6,p7,p8,p9,
                 p10,p11,p12,p13,p14,p15,p16,p17,
                 p18,p19,p20,p21,p22,p23,p24,p25,
                 p26,p27,p28,p29,p30,p31,p32,p33,
                 p34,p35,p36,p37,p38,p39,p40]
    

    #lookup for what the balls mean
    # outputs radlow, radhigh, Plow, Phigh
    ball_lookup = {
                    0: [0.0,0.0,0.0,0.0],
                    1: [0.8,1.25,0.8,2.],
                    2: [1.25,2.0,0.8,2.],
                    3: [2.0,4.0,0.8,2.],
                    4: [4.0,6.0,0.8,2.],
                    5: [6.0,22.,0.8,2.],
                    6: [0.8,1.25,2.,3.4],
                    7: [1.25,2.0,2.,3.4],
                    8: [2.0,4.0,2.,3.4],
                    9: [4.0,6.0,2.,3.4],
                    10: [6.0,22.,2.,3.4],
                    11: [0.8,1.25,3.4,5.9],
                    12: [1.25,2.0,3.4,5.9],
                    13: [2.0,4.0,3.4,5.9],
                    14: [4.0,6.0,3.4,5.9],
                    15: [6.0,22.,3.4,5.9],
                    16: [0.8,1.25,5.9,10.],
                    17: [1.25,2.0,5.9,10.],
                    18: [2.0,4.0,5.9,10.],
                    19: [4.0,6.0,5.9,10.],
                    20: [6.0,22.,5.9,10.],
                    21: [0.8,1.25,10.,17.],
                    22: [1.25,2.0,10.,17.],
                    23: [2.0,4.0,10.,17.],
                    24: [4.0,6.0,10.,17.],
                    25: [6.0,22.,10.,17.],
                    26: [0.8,1.25,17.,29.],
                    27: [1.25,2.0,17.,29.],
                    28: [2.0,4.0,17.,29.],
                    29: [4.0,6.0,17.,29.],
                    30: [6.0,22.,17.,29.],
                    31: [0.8,1.25,29.,50.],
                    32: [1.25,2.0,29.,50.],
                    33: [2.0,4.0,29.,50.],
                    34: [4.0,6.0,29.,50.],
                    35: [6.0,22.,29.,50.],
                    36: [0.8,1.25,50.,85.],
                    37: [1.25,2.0,50.,85.],
                    38: [2.0,4.0,50.,85.],
                    39: [4.0,6.0,50.,85.],
                    40: [6.0,22.,50.,85.],
                   }

    rsamps = random.choice(balls,size=nselect)
    radius = np.zeros(nselect)
    period = np.zeros(nselect)
    for i,samp in enumerate(rsamps):
        rl, rh, pl, ph = ball_lookup[samp]

        if samp in [5,10,15,20,25,30,35]:
        #check for giant planets
        # if a giant planet than draw from  a uniform in log
            radius[i] = np.e**random.uniform(low=np.log(rl),high=np.log(rh))
        else:
            radius[i] = random.uniform(low=rl,high=rh)

        period[i] = random.uniform(low=pl,high=ph)

    return radius, period


def TESS_noise_1h(mag):
    """
    returns noise in ppm for a transit of 1 hours duraton
    """
    mag_level, noise_level = np.genfromtxt('TessNoise_1h_v2.csv', delimiter=',',
                                           unpack=True,
                                           comments='#')

    # we probably shouldn't trust any extrapolated values
    # but it's all guess work at this stage anyway
    mag_interp = interp1d(mag_level, noise_level,
                          kind='cubic', fill_value='extrapolate')
    return mag_interp(mag)


# def isDetected(constants,starprop,planetprop):


def nearly_equal(a, b, sig_fig=3):
    return (a == b or
            int(a * 10**sig_fig) == int(b * 10**sig_fig)
            )







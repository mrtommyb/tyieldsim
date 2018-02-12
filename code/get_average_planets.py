import numpy as np
import pandas as pd
import astroquery
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from numpy.random import poisson, beta, uniform

import occSimFuncs as occFunc
from tvguide import TessPointing

from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm

msun = 1.9891E30
rsun = 695500000.
G = 6.67384E-11
AU = 149597870700.

consts = {'sigma_threshold': 7.3}
fn = '../data/cvz-south.csv.bz2'


tot_sum = np.zeros(10)
for i in tqdm(np.arange(0,10,1)):
    df = pd.read_csv(fn)
    df['isMdwarf'] = np.where((df.TEFF < 3900) & (df.RADIUS < 0.5), True, False)
    df['isGiant'] = np.ones(df.shape[0], dtype='bool') # assume all dwarfs

    df['cosi'] = pd.Series(np.random.random(size=df.shape[0]),name='cosi')

    df['noise_level'] = occFunc.component_noise(df.TESSMAG, readmod=1, zodimod=1)

    df['nsectors'] = np.round(df.obs_len/27.4)

    # how many planets should each star get
    np_fgk = poisson(lam=0.689,size=df.shape[0])
    np_m = poisson(lam=2.5,size=df.shape[0])
    df['Nplanets'] = pd.Series(np.where(df.isMdwarf, np_m, np_fgk), name='Nplanets')

    # draw a bunch of planest and accociate them with each star
    starID = 0 # ???
    newDF, starID = occFunc.make_allplanets_df_vec(df, starID)

    # get some transit epochs
    newDF = newDF.assign(T0=pd.Series(uniform(0, 1, size=newDF.shape[0]) * newDF.loc[:, 'planetPeriod']))

    #calculate the number of transits
    nt1 = np.floor(newDF['obs_len'] / newDF.planetPeriod)
    newDF['Ntransits'] = np.where(newDF['T0'] < newDF['obs_len'] %
        newDF.planetPeriod, nt1+1,nt1)
    newDF['ars'] = occFunc.per2ars(newDF.planetPeriod, newDF.MASS, newDF.RADIUS)
    newDF['ecc'] = pd.Series(beta(1.03,13.6,size=newDF.shape[0]),name='ecc', ) # ecc dist from Van Eylen 2015
    newDF['omega'] = pd.Series(uniform(-np.pi,np.pi,size=newDF.shape[0]),name='omega')
    newDF['rprs'] = occFunc.get_rprs(newDF.planetRadius, newDF.RADIUS)
    newDF['impact'] = newDF.cosi * newDF.ars * ((1-newDF.ecc**2)/1+newDF.ecc*np.sin(newDF.omega)) # cite Winn
    newDF['duration'] = occFunc.get_duration(newDF.planetPeriod, newDF.ars, cosi=newDF.cosi, b=newDF.impact,
                                    rprs=newDF.rprs) # cite Winn
    newDF['duration_correction'] = np.sqrt(newDF.duration * 24.) # correction for CDPP because transit dur != 1 hour
    newDF['transit_depth']  = occFunc.get_transit_depth(newDF.planetRadius, newDF.RADIUS)

    newDF['transit_depth_diluted']  = newDF['transit_depth'] / (1+newDF.CONTRATIO)

    newDF['needed_for_detection'] = (newDF.transit_depth_diluted * newDF.duration_correction *
                        np.sqrt(newDF.Ntransits)) / consts['sigma_threshold']

    newDF['has_transits']  = (newDF.ars > 1.0) & (newDF.impact < 1.0)

    newDF['detected'] = (newDF.noise_level < newDF.needed_for_detection) & (newDF.Ntransits >= 3) & (newDF.planetRadius > 0.0) & newDF.has_transits


    total_planets = newDF.detected.sum()
    tot_sum[i] = total_planets
    # print(total_planets)
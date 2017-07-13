import numpy as np
import pandas as pd
# import astroquery
# import matplotlib.pyplot as plt
# import glob
from tqdm import tqdm

# import matplotlib
from tvguide import TessPointing
from astropy.coordinates import SkyCoord
from astropy import units as u


def select_stars(df, selectnum, selectpoles):

    df.loc[:, 'SELECTED'] = np.zeros(df.shape[0], dtype='bool')

    eloop = np.linspace(0, 360, 14)
    espace = np.diff(eloop)[0]  # get space in degrees

    # poles first
    mn = (df.ECLAT >= 78)
    selectthese = df.loc[mn].sort_values('PRIORITY').iloc[-selectpoles:].index
    df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    mn = (df.ECLAT <= -78)
    selectthese = df.loc[mn].sort_values('PRIORITY').iloc[-selectpoles:].index
    df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    for elon in eloop[:-1]:
        # southern hemisphere
        ms = ((df.ECLONG >= elon) & (df.ECLONG < elon + espace) &
              (df.ECLAT <= -6) & (df.ECLAT > -78))
        selectthese = df.loc[ms].sort_values(
            'PRIORITY').iloc[-selectnum:].index
        df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    #     # northern hemisphere
        mn = ((df.ECLONG >= elon) & (df.ECLONG < elon + espace) &
              (df.ECLAT >= 6) & (df.ECLAT < 78))
        selectthese = df.loc[mn].sort_values(
            'PRIORITY').iloc[-selectnum:].index
        df.loc[df.index.isin(selectthese), 'SELECTED'] = True

        print(elon, elon + espace, df.loc[df.SELECTED].shape[0])
    return df[df.SELECTED]


def get_camera(df):
    # now add tvguide cameras
    df['tess_ccd'] = np.zeros(df.shape[0], dtype='int')
    df['obs_len'] = np.zeros(df.shape[0], dtype='float')
    for i in tqdm(df.index):
        # need a hack to force stars in the northern hemisphere to work
        if (~df.loc[i, 'SELECTED']) or (np.abs(df.loc[i, 'ECLAT']) < 6):
            continue
        elif (df.loc[i, 'SELECTED']) & (df.loc[i, 'ECLAT'] >= 6):
            gc = SkyCoord(lon=df.loc[i, 'ECLONG'] * u.degree,
                          lat=df.loc[i, 'ECLAT'] * u.degree * -1,
                          frame='barycentrictrueecliptic')
            obj = TessPointing(gc.icrs.ra.value, gc.icrs.dec.value)
        elif (df.loc[i, 'SELECTED']) & (df.loc[i, 'ECLAT'] <= -6):
            obj = TessPointing(df.loc[i, 'RA'], df.loc[i, 'DEC'])
        df.loc[i, 'tess_ccd'] = obj.get_camera()
        df.loc[i, 'obs_len'] = obj.get_13cameras()[obj.get_13cameras() >
                                                   0].shape[0] * 27.4
    return df


if __name__ == '__main__':
    fn = '/Users/tom/Dropbox/TIC5/ctl.csv'

    header = ['RA', 'DEC',
              'ECLONG', 'ECLAT', 'Ks',
              'TESSMAG', 'TEFF', 'RADIUS', 'MASS', 'CONTRATIO', 'PRIORITY',
              ]
    usecols = [13, 14, 60, 64, 87, 70, 72, 84, 26, 27, 46]
    df = pd.read_csv(fn, names=header, usecols=usecols)

    df = select_stars(df, selectnum=8000, selectpoles=6000)
    df = get_camera(df)

    selected = df[(df.SELECTED) & (df['obs_len'] > 0.0)]

    # selected.to_csv('../data/selectedreal5-MoreStars-v2.csv.bz2', compression='bz2')
    # selected.to_csv('../data/selectedreal5-LongerBaseline-v2.csv.bz2', compression='bz2')
    selected.to_csv('../data/selectedreal5-Nominal-v2.csv.bz2', compression='bz2')

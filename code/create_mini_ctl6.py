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
    mn = (df.ECLAT >= 74)
    selectthese = df.loc[mn].sort_values('PRIORITY').iloc[-selectpoles:].index
    df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    mn = (df.ECLAT <= -74)
    selectthese = df.loc[mn].sort_values('PRIORITY').iloc[-selectpoles:].index
    df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    for elon in eloop[:-1]:
        # southern hemisphere
        ms = ((df.ECLONG >= elon) & (df.ECLONG < elon + espace) &
              (df.ECLAT <= -6) & (df.ECLAT > -74))
        selectthese = df.loc[ms].sort_values(
            'PRIORITY').iloc[-selectnum:].index
        df.loc[df.index.isin(selectthese), 'SELECTED'] = True

    #     # northern hemisphere
        mn = ((df.ECLONG >= elon) & (df.ECLONG < elon + espace) &
              (df.ECLAT >= 6) & (df.ECLAT < 74))
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
    fn = '/Users/tom/Dropbox/TIC6/CTL6/all.csv'

    header = [
        'RA', 'DEC', 'TESSMAG', 'TEFF', 
        'PRIORITY', 'RADIUS', 'MASS', 'CONTRATIO', 
        'ECLONG', 'ECLAT', 'V', 'Ks', 'TICID',
    ]
    usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 33]



    print()
    print('doing nominal')
    df = pd.read_csv(fn, names=header, usecols=usecols)
    df = select_stars(df, selectnum=8200, selectpoles=6000)
    df = get_camera(df)
    selected = df[(df.SELECTED) & (df['obs_len'] > 0.0)]
    selected.to_csv('../data/selectedreal6-Nominal-v1.csv.bz2',
                    compression='bz2')

    print()
    print('doing longer baseline')
    df = pd.read_csv(fn, names=header, usecols=usecols)
    df = select_stars(df, selectnum=2200, selectpoles=12000)
    df = get_camera(df)
    selected = df[(df.SELECTED) & (df['obs_len'] > 0.0)]
    selected.to_csv('../data/selectedreal6-LongerBaseline-v1.csv.bz2',
                    compression='bz2')

    print()
    print('doing more stars')
    df = pd.read_csv(fn, names=header, usecols=usecols)
    df = select_stars(df, selectnum=11200, selectpoles=3000)
    df = get_camera(df)
    selected = df[(df.SELECTED) & (df['obs_len'] > 0.0)]
    selected.to_csv('../data/selectedreal6-MoreStars-v1.csv.bz2',
                    compression='bz2')

    print()
    print('doing entire CTL')
    df = pd.read_csv(fn, names=header, usecols=usecols)
    df = select_stars(df, selectnum=1000000000, selectpoles=1000000000)
    df = get_camera(df)
    selected = df[(df.SELECTED) & (df['obs_len'] > 0.0)]
    selected.to_csv('../data/allCTL6-v1.csv.bz2',
                    compression='bz2')




    # print()
    # print('doing cvz south')
    # df = pd.read_csv(fn, names=header, usecols=usecols)
    # df = df.loc[df.ECLAT <= -78]
    # df.loc[:,'SELECTED'] = True
    # selected = get_camera(df)
    # selected.to_csv('../data/cvz-south.csv.bz2', compression='bz2')

    # print()
    # print('doing cvz south plus')
    # df = pd.read_csv(fn, names=header, usecols=usecols)
    # df = df.loc[df.ECLAT <= -73]
    # df.loc[:,'SELECTED'] = True
    # selected = get_camera(df)
    # selected.to_csv('../data/cvz-south-plus.csv.bz2', compression='bz2')

    # print()
    # print('doing cvz north')
    # df = pd.read_csv(fn, names=header, usecols=usecols)
    # df = df.loc[df.ECLAT >= 78]
    # df.loc[:,'SELECTED'] = True
    # selected = get_camera(df)
    # selected.to_csv('../data/cvz-north.csv.bz2', compression='bz2')

    # print()
    # print('doing cvz north plus')
    # df = pd.read_csv(fn, names=header, usecols=usecols)
    # df = df.loc[df.ECLAT >= 73]
    # df.loc[:,'SELECTED'] = True
    # selected = get_camera(df)
    # selected.to_csv('../data/cvz-north-plus.csv.bz2', compression='bz2')

    # selected.to_csv('../data/selectedreal5-MoreStars-v2.csv.bz2', compression='bz2')
    # selected.to_csv('../data/selectedreal5-LongerBaseline-v2.csv.bz2', compression='bz2')

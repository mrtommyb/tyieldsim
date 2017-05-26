
# coding: utf-8
# Tom Barclay tom@tombarlay.com
# # Simulate exoplanet yield from TESS
# The purpose of this code is to simulate the exoplanet yield from the TESS Mission.
# We do this by taking the various fields that TESS observes and, using a galaxy model, 
#   put planets orbiting the stars and see whether we can detect that planet.


from __future__ import division, print_function
import numpy as np
import pandas as pd
import astroquery
import matplotlib.pyplot as plt
import glob
import matplotlib
matplotlib.style.use('ggplot')

import occSimFuncs as occFunc
from numpy.random import poisson, beta, uniform



#get_ipython().magic(u'matplotlib inline')

#constants
msun = 1.9891E30
rsun = 695500000.
G = 6.67384E-11
AU = 149597870700.


# lets read our galaxt model files

def add_stellar_props(df):
    df['isMdwarf'] = pd.Series((df.CL == 5) & (df.Typ >= 7.), name='isMdwarf' )
    df['isGiant'] = pd.Series((df.CL < 5), name='isGiant' )
    df['I'] = pd.Series(-1. * (df.VI - df.V), name='I')
    df['Teff'] = pd.Series(10**df.LTef , name='Teff')

    g = 10**df.logg * 0.01
    df['Radius'] = pd.Series(np.sqrt(G*df.Mass*msun / g) / rsun, name='Radius')

    return df

def make_fullfov_df(df,consts):
    multiple = consts['galmodarea'] / consts['simsize']
    numstars = int(df.shape[0] * multiple)
    rows = np.random.choice(df.index.values, size=numstars)
    newq = df.ix[rows]
    
    return newq.set_index(np.arange(newq.shape[0]))

def make_allplanets_df(df,starid_zp):
    df['planetRadius'] = pd.Series()
    df['planetPeriod'] = pd.Series()
    df['starID'] = pd.Series()

    newdf = pd.DataFrame(columns=df.columns)
    starID = starid_zp
    for thisRow in np.arange(df.shape[0]):
        if df.loc[thisRow,'isMdwarf']:
            radper = occFunc.Dressing15_select(df.loc[thisRow, 'Nplanets'])
            if df.loc[thisRow,'Nplanets'] == 0:
                continue
            elif df.loc[thisRow,'Nplanets'] == 1:
                df.loc[thisRow,'planetRadius'] = radper[0]
                df.loc[thisRow,'planetPeriod'] = radper[1]
                df.loc[thisRow,'starID'] = starID
                newdf = newdf.append(df.loc[thisRow])
                starID +=1
            elif df.loc[thisRow,'Nplanets'] >= 2:
                df.loc[thisRow,'starID'] = starID
                for p in np.arange(df.loc[thisRow,'Nplanets']):
                    df.loc[thisRow,'planetRadius'] = radper[0][p]
                    df.loc[thisRow,'planetPeriod'] = radper[1][p]
                    newdf = newdf.append(df.loc[thisRow])
                starID +=1
            
        elif not df.loc[thisRow,'isMdwarf']:
            radper = occFunc.Fressin13_select(df.loc[thisRow, 'Nplanets'])
            if df.loc[thisRow,'Nplanets'] == 0:
                continue
            elif df.loc[thisRow,'Nplanets'] == 1:
                df.loc[thisRow,'planetRadius'] = radper[0]
                df.loc[thisRow,'planetPeriod'] = radper[1]
                df.loc[thisRow,'starID'] = starID
                newdf = newdf.append(df.loc[thisRow])
                starID +=1
            elif df.loc[thisRow,'Nplanets'] >= 2:
                df.loc[thisRow,'starID'] = starID
                for p in np.arange(df.loc[thisRow,'Nplanets']):
                    df.loc[thisRow,'planetRadius'] = radper[0][p]
                    df.loc[thisRow,'planetPeriod'] = radper[1][p]
                    newdf = newdf.append(df.loc[thisRow])    
                starID += 1

    newdf.set_index(np.arange(newdf.shape[0]), inplace=True)

    return newdf, starID

def make_allplanets_df_vec(df,starid_zp):
    # lets refector the above code to make it array operations
    totalRows = df['Nplanets'].sum()

    df['planetRadius'] = pd.Series()
    df['planetPeriod'] = pd.Series()
    df['starID'] = pd.Series()

    radper_dressing = occFunc.Dressing15_select(totalRows)
    radper_fressin = occFunc.Fressin13_select(totalRows)

    #we need an array of indices
    totalRows = df['Nplanets'].sum()
    rowIdx = np.repeat(np.arange(df.shape[0]),np.array(df.Nplanets.values))

    newdf = df.iloc[rowIdx]
    newdf.starID = rowIdx + starid_zp


    newdf.planetRadius = np.where(newdf.isMdwarf,radper_dressing[0],radper_fressin[0])
    newdf.planetPeriod = np.where(newdf.isMdwarf,radper_dressing[1],radper_fressin[1])
    newdf.set_index(np.arange(newdf.shape[0]), inplace=True)

    return newdf, newdf.starID.iloc[-1]


if __name__ == '__main__':

    # the besancon galaxy file column names
    names = ['Dist','Mv','CL','Typ','LTef','logg','Age',
             'Mass','BV','UB','VI','VK','V','FeH',
             'l','b','Av','Mbol']

    # some constants we need
    consts =  {'sigma_threshold': 10.,
               'simsize': 8, #size of the galmod field in sq deg
                'full_fov': True, # if true do whole 24x24deg ccd
              }

    galmodfiles = glob.glob('../data/besmod*.csv')


    #loop over every field
    starID = 0
    sumq = 0

    for thisGalmodidx in range(len(galmodfiles)):

        consts['galmodarea'] = np.genfromtxt('bess_reas.txt',usecols=2)[thisGalmodidx]
        consts['ra'] = np.genfromtxt('bess_reas.txt',usecols=0)[thisGalmodidx]
        consts['dec'] = np.genfromtxt('bess_reas.txt',usecols=1)[thisGalmodidx]
        consts['tess_ccd'] = np.genfromtxt('bess_reas.txt',usecols=3)[thisGalmodidx]

        intial_q = pd.read_csv(galmodfiles[thisGalmodidx], skiprows=2, names=names)
        intial_q = add_stellar_props(intial_q)

        # we previously got the besancon models, they are in the directory ../data/
        # We also saved the areas for each field in bess_reas.txt

        # We are doing this in a monte carlo fashion. Our outer loop is each field. 
        # The row closest to the equator is easiest because there is no overlap.
        # We saved a few functions in occSimFuncs.py

        #make the catalog equal to the full fov area
        if consts['full_fov'] & (consts['simsize'] < consts['galmodarea']):
            q = make_fullfov_df(intial_q,consts)
            
        elif (simsize > consts['galmodarea']):
            raise('Galmod area is too small!')

        else:
            q = intial_q

        # here is how we calculate the observations length
        # - all field on TESS ccd 1 will have an obselen of 27 days
        # - targets on CCD 2 will have either a 54 days or 27 days
        #        - fraction with 54 days x = ((24**2) - A) / (2*A)
        #        - fraction with 27 days = A* (1-x)
        # - targets of CCD are harder
        # for now let's do the same things as CCD2 but THIS IS NOT RIGHT
        #   because some of the targets have obslen = 81 days
        # - targets on CCD 4 have obslen = 351

        q['ra'] = np.zeros(q.shape[0]) + consts['ra']
        q['dec'] = np.zeros(q.shape[0]) + consts['dec']
        q['tess_ccd'] = np.zeros(q.shape[0]) + consts['tess_ccd']
        if consts['tess_ccd'] == 1:
            consts['obslen'] = 27 # the days we observe field, needs to change
            q['obslen'] = np.zeros(q.shape[0]) + consts['obslen']
        elif (consts['tess_ccd'] == 2) or (consts['tess_ccd'] == 3):
            consts['obslen'] = [27,54]
            x_54 = ((24**2) - consts['galmodarea']) / (2*consts['galmodarea'])
            x_27 = consts['galmodarea'] * (1-x_54)
            idx_54 = np.random.choice(np.arange(q.shape[0]), size=x_54*q.shape[0], replace=False)
            idx_54_bool = np.in1d(np.arange(q.shape[0]),idx_54)
            q['obslen'] = np.where(idx_54_bool,54,27)
        elif consts['tess_ccd'] == 4:
            consts['obslen'] = 351
            q['obslen'] = np.zeros(q.shape[0]) + consts['obslen']



        #some planet parameters we will need later
        q['cosi'] = pd.Series(np.random.random(size=q.shape[0]),name='cosi')
        # this cosi will be the same for every planet in the system

        q['noise_level'] = occFunc.TESS_noise_1h(q.I)

        np_fgk = poisson(lam=0.689,size=q.shape[0])
        np_m = poisson(lam=2.5,size=q.shape[0])
        q['Nplanets'] = pd.Series(np.where(q.isMdwarf, np_m,np_fgk), name='Nplanets')


        # draw a bunch of planest and accociate them with each star
        newDF, starID = make_allplanets_df_vec(q, starID)


        newDF['T0'] = uniform(0,1,size=newDF.shape[0]) * newDF.planetPeriod

        nt1 = np.floor((newDF['obslen']-newDF.T0) / newDF.planetPeriod)

        newDF['Ntransits'] = np.where(newDF['obslen'] <= newDF.planetPeriod, nt1+1,0)
        newDF['ars'] = occFunc.per2ars(newDF.planetPeriod,newDF.Mass,newDF.Radius)
        newDF['ecc'] = pd.Series(beta(1.03,13.6,size=newDF.shape[0]),name='ecc', ) # ecc dist from Van Eylen 2015
        newDF['omega'] = pd.Series(uniform(-np.pi,np.pi,size=newDF.shape[0]),name='omega')
        newDF['rprs'] = occFunc.get_rprs(newDF.planetRadius,newDF.Radius)
        newDF['impact'] = newDF.cosi * newDF.ars * ((1-newDF.ecc**2)/1+newDF.ecc*np.sin(newDF.omega)) # cite Winn
        newDF['duration'] = occFunc.get_duration(newDF.planetPeriod,newDF.ars,cosi=newDF.cosi, b=newDF.impact,
                                        rprs=newDF.rprs) # cite Winn
        newDF['duration_correction'] = np.sqrt(newDF.duration * 24.) # correction for CDPP because transit dur != 1 hour
        newDF['transit_depth']  = occFunc.get_transit_depth(newDF.planetRadius,newDF.Radius)


        # now lets see if those planets are detected

        newDF['needed_for_detection'] = (newDF.transit_depth * newDF.duration_correction *
                    np.sqrt(newDF.Ntransits)) / consts['sigma_threshold']
        newDF['has_transits']  = (newDF.ars > 1.0) & (newDF.impact < 1.0)


        newDF['detected'] = (newDF.noise_level < newDF.needed_for_detection) & (newDF.Ntransits >= 3) & (newDF.planetRadius > 0.0) & newDF.has_transits


        total_planets = newDF.detected.sum()

        try:
            detected_DF = pd.concat([detected_DF,newDF[newDF.detected == True]])
        except:
            detected_DF = newDF[newDF.detected == True]
        

        sumq += q.shape[0]



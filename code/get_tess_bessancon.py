from __future__ import division, print_function
import numpy as np
import pandas as pd
import astroquery
import matplotlib.pyplot as plt
from astroquery.besancon import Besancon
import ephem


lon = np.repeat(np.linspace(0,360,14)[:-1],2)
lat = np.tile([18,-18],13)
centers_row1 = np.array([lon,lat]) # long/lat in degrees
areas_row1 = np.repeat(24*24,26) # area in sq deg = 24*24

lat = np.tile([42,-42],13)
centers_row2 = np.array([lon,lat]) # long/lat in degrees

#There are again 13 rows but this time we want to think carefully about the areas because there are overlaps.
#The area of the spherical zone can be calculated using the area of a spherical cap as 
#A = 2*pi*(1- cos( 90-lat ) )
#So... area of the how segment is

A = (2*np.pi*(1 - np.cos(np.radians(90 - (42-12)))) 
     - 2*np.pi*(1 - np.cos(np.radians(90 - (42+12) )  )))

areas_row2 = np.repeat((A * (180/np.pi)**2)/13,26)

lat = np.tile([66,-66],13)
centers_row3 = np.array([lon,lat]) # long/lat in degrees


A = (2*np.pi*(1 - np.cos(np.radians(90 - (66-12)))) 
     - 2*np.pi*(1 - np.cos(np.radians(90 - (66+12) )  )))
areas_row3 = np.repeat((A * (180/np.pi)**2)/13,26)

centers_row4 = np.array([[0,0],[90,-90]]) # long/lat in degrees

areas_row4 = np.repeat(24*24,2)

####

maglim = {'U':(-99, 99), 'B':(-99, 99), 'V':(-99, 99), 'R':(-99, 99),
          'I':(4, 13.5), 'J':(-99, 99), 'H':(-99, 99), 'K':(-99, 99),
          'L':(-99, 99)}

absmaglim = (-7, 20)

for center in np.r_[centers_row1.T,centers_row2.T,centers_row3.T,centers_row4.T]:
    thisEcl = center
    ecl_obj = ephem.Ecliptic(*np.radians(thisEcl))
    gal_obj = ephem.Galactic(ecl_obj)

    model = Besancon.query(glon=np.float(np.degrees(gal_obj.lon)), 
						   glat=np.float(np.degrees(gal_obj.lat)),
					       email='tom@tombarclay.com',
					       area=8.0,
					       mag_limits=maglim,
					       absmag_limits=absmaglim,
					       verbose=False,
					       )
    
    model.write('besmod_{}_{}.csv'.format(*thisEcl), format='ascii', delimiter=',')




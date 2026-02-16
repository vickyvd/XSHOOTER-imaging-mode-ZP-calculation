#Stetson catalog has bands UVBRI and their errors and we are going to fill the sdss u'g'r'i'z' bands using Jester 2005.
from astropy.table import Table

#we load the catalog: magnitudes and errors
cat_stetson = Table.read("stetson_cat.csv", format="csv")

Umag = cat_stetson['Umag']
Bmag = cat_stetson['Bmag']
Vmag = cat_stetson['Vmag']
Rmag = cat_stetson['Rmag']
Imag = cat_stetson['Imag']
e_Umag = cat_stetson['e_Umag']
e_Bmag = cat_stetson['e_Bmag']
e_Vmag = cat_stetson['e_Vmag']
e_Rmag = cat_stetson['e_Rmag']
e_Imag = cat_stetson['e_Imag']

#we apply the Jester 2005 transformations to get the u'g'r'i'z' magnitudes and errors
g_prime = Vmag + 0.60*(Bmag - Vmag) - 0.12
e_g_prime = (e_Vmag**2 + (0.60*e_Bmag)**2 + (0.60*e_Vmag)**2)**0.5
r_prime = Vmag - 0.42*(Bmag - Vmag) + 0.11
e_r_prime = (e_Vmag**2 + (0.42*e_Bmag)**2 + (0.42*e_Vmag)**2)**0.5
u_prime = g_prime + 1.28*(Umag - Bmag) + 1.13
e_u_prime = (e_g_prime**2 + (1.28*e_Umag)**2 + (1.28*e_Bmag)**2)**0.5
i_prime = r_prime - 0.91*(Bmag - Vmag) + 0.2
e_i_prime = (e_r_prime**2 + (0.91*e_Bmag)**2 + (0.91*e_Vmag)**2)**0.5
z_prime = r_prime - 1.72*(Bmag - Vmag) + 0.41
e_z_prime = (e_r_prime**2 + (1.72*e_Bmag)**2 + (1.72*e_Vmag)**2)**0.5

#we save the results in a new table
cat_stetson['u_primemag'] = u_prime
cat_stetson['e_u_primemag'] = e_u_prime
cat_stetson['g_primemag'] = g_prime
cat_stetson['e_g_primemag'] = e_g_prime
cat_stetson['r_primemag'] = r_prime
cat_stetson['e_r_primemag'] = e_r_prime
cat_stetson['i_primemag'] = i_prime
cat_stetson['e_i_primemag'] = e_i_prime
cat_stetson['z_primemag'] = z_prime
cat_stetson['e_z_primemag'] = e_z_prime

#we save the new table in a csv file
cat_stetson.write("stetson_cat_final.csv", format="csv", overwrite=True)
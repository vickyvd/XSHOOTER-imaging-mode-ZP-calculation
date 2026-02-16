#Import necessary libraries
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from astropy.stats import sigma_clipped_stats
from astropy.io import fits 
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import ccdproc
from astropy.nddata import CCDData
import glob
import scipy.optimize as opt
import yaml
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


def zp_estimation(global_path, filter):

    with open(f'{global_path}config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_path = f'{config["paths"]["data"]}'

    #WE LOAD THE DATA: science frames, skyff frames and bias frames.

    filter_kward = config['header_kwards']['filter']
    frame_kward =  config['header_kwards']['frame']

    science_paths = []
    skyff_paths = []
    bias_paths = []

    for file in glob.glob(global_path + data_path + '*'):
        header = fits.getheader(file)
        if header[frame_kward] == config['others']['science_frame'] and header[filter_kward] == filter:
            science_paths.append(file)
        elif header[frame_kward] == config['others']['skyff_frame'] and header[filter_kward] == filter:
            skyff_paths.append(file)
        elif header[frame_kward] == config['others']['bias_frame']:
            bias_paths.append(file)

    science_paths.sort() #so we can use as reference wcs the first exposure (center)
    print('Science frames files:')
    for file in science_paths:
        print(file)
    print('------------------')

    print('Skyff frames files:')
    for file in skyff_paths:
        print(file)
    print('------------------')

    print('Bias frames files:')
    for file in bias_paths:
        print(file)
    print('------------------')

    #WE REDUCE THE DATA: we create master bias and master skyff, and we reduce the science frames by subtracting the master bias and dividing by the master skyff.

    #we save a header to extract keyword values later
    ref_header = fits.getheader(science_paths[0])
    #we load importan paramters
    gain = ref_header[config['header_kwards']['gain']]
    pix_scale = config['others']['pix_scale']
    dither = config['others']['dither']
    overscan_x1 = config['others']['overscan_x'][0]
    overscan_x2 = config['others']['overscan_x'][1]
    overscan_y1 = config['others']['overscan_y'][0]
    overscan_y2 = config['others']['overscan_y'][1]
    seeing_start = ref_header[config['header_kwards']['seeing_start']]
    seeing_end = ref_header[config['header_kwards']['seeing_end']]
    fwhm = (seeing_start + seeing_end)/2
    airmass_start = ref_header[config['header_kwards']['airmass_start']]
    airmass_end = ref_header[config['header_kwards']['airmass_end']]
    airmass = (airmass_start + airmass_end)/2
    aper_radius = config['others']['aper_radius']*fwhm/pix_scale #aperture radius in pixels
    inner_annulus_radius = config['others']['iner_annulus_radius']*fwhm/pix_scale #inner radius of the annulus for background estimation in pixels
    outer_annulus_radius = config['others']['outer_annulus_radius']*fwhm/pix_scale #outer radius of the annulus for background estimation in pixels
    bin_width = config['others']['bin_width'] #pixel size of bins for histogram of shifts between detected and catalog sources
    good_dist = config['others']['good_dist'] #maximum distance in pixels to consider a detected source as a good match to a catalog source
    detect_thresh = config['others']['detect_thresh'][filter] #threshold for source detection in units of the background std
    readout_noise = config['others']['readout_noise'] #in e-
    exptime = ref_header[config['header_kwards']['exptime']]

    ccd_science = []
    wcs = []
    for path in science_paths:
        data, header = fits.getdata(path, header=True)
        wcs_data = WCS(header)
        wcs.append(wcs_data)
        ccd_sci = CCDData(data[overscan_y1:overscan_y2, overscan_x1:overscan_x2], meta=header, unit=u.adu, wcs=wcs_data)
        ccd_sci = ccdproc.gain_correct(ccd_sci, gain*u.electron/u.adu) #In e-
        ccd_science.append(ccd_sci)
    wcs_data = wcs[0].slice((slice(overscan_y1, overscan_y2), slice(overscan_x1, overscan_x2)))

    master_bias = ccdproc.combine(bias_paths, unit=u.adu, method='median') #MASTER BIAS
    bias = ccdproc.gain_correct(master_bias, gain*u.electron/u.adu) #In e-
    bias = bias[overscan_y1:overscan_y2, overscan_x1:overscan_x2]

    ccd_skyff = []
    for path in skyff_paths:
        data, header = fits.getdata(path, header=True)
        skyff = CCDData(data[overscan_y1:overscan_y2, overscan_x1:overscan_x2], meta=header, unit=u.adu)
        skyff = ccdproc.gain_correct(skyff, gain*u.electron/u.adu) #In e-
        skyff_sub_bias = ccdproc.subtract_bias(skyff, bias) #subtract master bias from each skyff frame
        ccd_skyff.append(skyff_sub_bias)

    #master skyff combined using a sigma-clipped average and we normalize it by its mean.
    skyff_sub_bias = ccdproc.combine(ccd_skyff, method='average', sigma_clip=True, sigma_clip_high_thresh=3, sigma_clip_low_thresh=3, sigma_clip_func=np.ma.median) 
    skyff_norm = skyff_sub_bias / np.nanmean(skyff_sub_bias.data)
    skyff_norm = CCDData(skyff_norm.data, meta=skyff_sub_bias.meta, unit=skyff_sub_bias.unit)

    ccd_science_final = []
    for i, sci in enumerate(ccd_science):
        sci_sub_bias = ccdproc.subtract_bias(sci, bias) #subtract master bias from each science frame
        sci_flat_corr = ccdproc.flat_correct(sci_sub_bias, skyff_norm) #divide each science frame by the normalized master skyff
        sci_final = ccdproc.cosmicray_lacosmic(sci_flat_corr, sigclip=5) #remove cosmic rays 
        ccd_science_final.append(sci_final)
    #align all frames into the same WCS reference frame (the one of the first science frame)
    reprojected_ccd_science = []
    for i in range(len(ccd_science_final)):
        reprojected_ccd_science.append(ccdproc.wcs_project(ccd_science_final[i], target_wcs=wcs_data))

    #combine the aligned science frames using a sigma-clipped average 
    combiner = ccdproc.Combiner(reprojected_ccd_science)
    science_image_final = combiner.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median)
    science_image_final = combiner.average_combine()

    #we trim 17 pix ~ 3'' (to avoid dithering edges)
    pix_trim = int(np.floor(dither/pix_scale))
    science_image_final = science_image_final[pix_trim:science_image_final.shape[0]-pix_trim, pix_trim:science_image_final.shape[1]-pix_trim]
    wcs_data = wcs_data.slice((slice(pix_trim, science_image_final.shape[0]-pix_trim), slice(pix_trim, science_image_final.shape[1]-pix_trim)))

    #WE DETECT THE SOURCES FOR THE PHOTOMETRY

    #We load table 2 from Pancino 2022 which corresponds to Stetsons L98SA field (extension of Landolt's SA98 field) and contains the magnitudes of the stars in the field in different filters. 
    cat_stetson = Table.read("stetson_cat_final.csv", format="csv")
    ra_deg = cat_stetson['RAJ2000_deg']
    dec_deg = cat_stetson['DEJ2000_deg']
    coords_cat = SkyCoord(ra_deg*u.degree, dec_deg*u.degree, frame='icrs')
    x_cat, y_cat = wcs_data.world_to_pixel(coords_cat)

    #source detection 
    mean, median, std = sigma_clipped_stats(science_image_final.data)
    daofind = DAOStarFinder(fwhm=fwhm/pix_scale, threshold=detect_thresh*std)
    sources = daofind(np.array(science_image_final.data) - median)
    #we look for sources at aper_radius from the edges (so that we can do aperture photometry without worrying about the edges)
    mask = (sources['xcentroid'] > aper_radius) & (sources['xcentroid'] < science_image_final.shape[1] - aper_radius) & (sources['ycentroid'] > aper_radius) & (sources['ycentroid'] < science_image_final.shape[0] - aper_radius)
    sources = sources[mask]

    #There is a shift between the catalog coords and the detected sources coords, so we look for the shift in x and y by looking at the histogram of the differences in x and y between the detected sources and the catalog sources. We look for the meadian of the peak in the histogram to find the most common shift in x and y, which corresponds to the shift between the catalog and the detected sources.
    x_det = sources['xcentroid']
    y_det = sources['ycentroid']
    #differences in x and y between the detected sources and the catalog sources
    dx = x_det[:, None] - x_cat[None, :]
    dy = y_det[:, None] - y_cat[None, :]
    dx = dx.ravel()
    dy = dy.ravel()
    bins_dx = np.arange(dx.min(), dx.max() + bin_width, bin_width)
    hist_dx, edges_dx = np.histogram(dx, bins=bins_dx)
    imax_dx = np.argmax(hist_dx) #bin with the most common shift in x
    dx_min = edges_dx[imax_dx]
    dx_max = edges_dx[imax_dx + 1]
    bins_dy = np.arange(dy.min(), dy.max() + bin_width, bin_width)
    hist_dy, edges_dy = np.histogram(dy, bins=bins_dy)
    imax_dy = np.argmax(hist_dy) #bin with the most common shift in y
    dy_min = edges_dy[imax_dy]
    dy_max = edges_dy[imax_dy + 1]
    dx_peak = dx[(dx >= dx_min) & (dx < dx_max)]
    dy_peak = dy[(dy >= dy_min) & (dy < dy_max)]
    dx_shift = np.median(dx_peak) #median within the peak bin in x
    dy_shift = np.median(dy_peak) #median within the peak bin in y
    #we apply the shift to the catalog coordinates to match them with the detected sources coordinates
    x_cat_shift = x_cat + dx_shift
    y_cat_shift = y_cat + dy_shift
    #we look for the closest detected source to each catalog source using a KDTree and we keep only the good matches 
    tree = cKDTree(np.column_stack((x_det, y_det)))
    dist, idx = tree.query(np.column_stack((x_cat_shift, y_cat_shift)), k=1)
    good = dist < good_dist
    sources_matched = sources[idx[good]]
    sources_matched = sources_matched[np.argsort(idx[good])] #sort them by id

    #PHOTOMETRY ON THE DETECTED SOURCES
    x, y = sources_matched['xcentroid'], sources_matched['ycentroid']
    aperture = CircularAperture([(xi,yi) for xi, yi in zip(x, y)], r=aper_radius)
    annulus = CircularAnnulus([(xi,yi) for xi, yi in zip(x, y)], r_in=inner_annulus_radius, r_out=outer_annulus_radius)
    ra, dec = wcs_data.wcs_pix2world(x, y, 0)
    sources_matched['ra'] = ra
    sources_matched['dec'] = dec

    #we estimate the background
    a_mask_b = annulus.to_mask(method='center')

    mean_local =[]
    bkg_local = []
    std_bkg_local = []

    for mask in a_mask_b: 
        a_data_1d = mask.get_values(science_image_final)
        mean, median, std = sigma_clipped_stats(a_data_1d, sigma = 4) 
        bkg_local.append(median)
        std_bkg_local.append(std)
        mean_local.append(mean)
        
    bkg_local = np.array(bkg_local)
    std_bkg_local = np.array(std_bkg_local)
    mean_local = np.array(mean_local)
    mean_local, bkg_local, std_bkg_local

    #aperture photometry and flux estimation
    photometry = aperture_photometry(science_image_final, aperture)
    photometry['aperture_sum_bkgsub'] = photometry['aperture_sum'].value - bkg_local * aperture.area
    photometry['aperture_sum_bkgsub_error'] = np.sqrt(photometry['aperture_sum_bkgsub'].value + aperture.area*(mean_local + readout_noise**2)) 
    #we estimate the extinction using Patat 2011
    lambda_A = np.array(config['others']['lambda_A'])
    k_lambda = np.array(config['others']['k_lambda'])
    k_interp = interp1d(lambda_A, k_lambda, kind='linear', bounds_error=False, fill_value='extrapolate')
    pivot_wavelength = config['others']['pivot_wavelength'][filter] #in Angstroms
    atm_ext = k_interp(pivot_wavelength) * airmass

    #we estimate the instrumental magnitude
    ins_mag = np.array(-2.5 * np.log10(photometry['aperture_sum_bkgsub'].value/exptime)) 
    ins_mag_error = np.array((2.5 / np.log(10)) * (photometry['aperture_sum_bkgsub_error'].value / photometry['aperture_sum_bkgsub'].value))

    # WE ESTIMATE THE ZP

    #We load the needed filter mag and color from the Stetson catalog
    mag_cat = []
    mag_cat_error = []
    color_cat = []
    color_cat_error = []
    color_index = config['others']['color_index'][filter] #color to use for the ZP estimation, for example 'B-V' or 'g_prime-r_prime'

    for i in range(len(sources_matched)):
        x_match = sources_matched['xcentroid'][i]
        y_match = sources_matched['ycentroid'][i]
        idx_cat = np.argmin(np.sqrt((x_cat_shift - x_match)**2 + (y_cat_shift - y_match)**2))
        mag_cat.append(cat_stetson[filter+'mag'][idx_cat])
        mag_cat_error.append(cat_stetson['e_'+filter+'mag'][idx_cat])
        color_cat.append(cat_stetson[color_index[0]][idx_cat]-cat_stetson[color_index[1]][idx_cat])
        color_cat_error.append(np.sqrt(cat_stetson['e_'+color_index[0]][idx_cat]**2 + cat_stetson['e_'+color_index[1]][idx_cat]**2))

    mag_cat = np.array(mag_cat)
    mag_cat_error = np.array(mag_cat_error)
    color_cat = np.array(color_cat)
    color_cat_error = np.array(color_cat_error)

    #we estimate the differences between out photometry and the catalog's
    delta_mag = mag_cat - ins_mag
    delta_mag_error = np.sqrt(ins_mag_error**2 + mag_cat_error**2)

    #we fit the model, we sigmaclip data points that are more than 3 sigma away from the model and we repeat this process 5 times to get a better estimation of the ZP and the color term. 

    def model(color_index, zp, ct):
        return zp + ct * color_index - atm_ext

    color_cat_sig = color_cat
    delta_mag_sig = delta_mag   
    delta_mag_error_sig = delta_mag_error
    for _ in range(5):
        popt, pcov = opt.curve_fit(
            model,
            color_cat_sig,
            delta_mag_sig,
            sigma=delta_mag_error_sig,
            absolute_sigma=True
        )
        resid = delta_mag_sig - model(color_cat_sig, *popt)
        chi = resid / delta_mag_error_sig
        mask = np.abs(chi) < 10

        print(resid, delta_mag_error_sig, chi)

        color_cat_sig = color_cat_sig[mask]
        delta_mag_sig = delta_mag_sig[mask]
        delta_mag_error_sig = delta_mag_error_sig[mask]


    zp, ct = popt
    zp_error, ct_error = np.sqrt(np.diag(pcov))

    print(f'Results for filter {filter}: ZP = {zp:.3f} +/- {zp_error:.3f}, Color term = {ct:.3f} +/- {ct_error:.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    norm = simple_norm(science_image_final, 'linear', percent=99)
    im = ax[0].imshow(science_image_final, norm=norm, origin='lower', cmap='gray')
    plt.colorbar(im, ax=ax[0], shrink=0.75, label='e-')
    for i in range(len(sources_matched)):
        ax[0].text(sources_matched['xcentroid'][i]+10, sources_matched['ycentroid'][i]+10, str(i+1), color='orange', fontsize=8)
    aperture.plot(color='red', lw=1.5, alpha=0.5, ax =ax[0])
    annulus.plot(color='blue', lw=1.5, alpha=0.5, ax=ax[0])
    ax[0].set_ylabel(' ',fontsize = 10)
    ax[0].set_xlabel(' ',fontsize = 10)
    ax[1].plot(color_cat_sig, 1*zp + ct*color_cat_sig-atm_ext, color='orange', label=f'ZP = {zp:.3f} +/- {zp_error:.3f} \nCT = {ct:.3f} +/- {ct_error:.3f}', zorder=4)
    ax[1].errorbar(color_cat_sig, delta_mag_sig, yerr=delta_mag_error_sig, fmt='o', color='blue', elinewidth=1, capsize=2, zorder=5)
    ax[1].errorbar(color_cat, delta_mag,yerr=delta_mag_error,fmt='o', color='red', elinewidth=1, capsize=2)
    for i, label in enumerate([i for i in range(len(delta_mag))]):
        ax[1].text(color_cat[i]+0.02, delta_mag[i]+0.02, label+1, fontsize=8, color='red', clip_on=True)
    ax[1].set_xlabel(f"{color_index[0]} - {color_index[1]} (mag)")
    ax[1].set_ylabel(f"Catalog {filter} - Instrumental mag (mag)")
    plt.legend()
    plt.show()

    return zp, zp_error, ct, ct_error


#MAIN: logic to estimate all the ZP for the 10 filters and to save the results in a file. 
def main():
    global_path = '/home/victoriavd/xshooter_2026/' #MAYBE ADD IT AS USER INPUT
    filters = ['U', 'B', 'V', 'R', 'I', 'u_prime', 'g_prime', 'r_prime', 'i_prime', 'z_prime']
    table_results = Table(names=['filter', 'ZP', 'ZP_error', 'Color_term', 'Color_term_error'], dtype=['str', 'float', 'float', 'float', 'float'])
    for filter in filters:
        zp, zp_error, ct, ct_error = zp_estimation(global_path, filter)
        table_results.add_row([filter, zp, zp_error, ct, ct_error])
    table_results.write(global_path+'zp_results.csv', format='csv', overwrite=True)
if __name__ == "__main__":    
    main()
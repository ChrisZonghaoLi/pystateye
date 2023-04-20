import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
plt.style.use(style='default')
plt.rcParams['font.family']='calibri'

import time

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)

def statistical_eye(pulse_response, 
                    samples_per_symbol=8, 
                    M=4, # 2 for NRZ and 4 for PAM4
                    vh_size=2048, # vertical voltage discretized level, has be even
                    A_window_multiplier=2, # control the vertical viewing space of the plot
                    sample_size=32, 
                    mu_noise=0, # V
                    sigma_noise=1.33e-4, # V 
                    mu_jitter=0.0125, # in terms of UI
                    sigma_jitter=0.015, # in terms of UI
                    target_BER=2.4e-4,
                    noise_flag=False,
                    jitter_flag=False,
                    plot=False,
                    pdf_conv_flag=True, # if you want to do pdf convolution to find all ISI conbinations, False will then brute force find all combination
                    diff_signal=True, # eye diagram amplitude will be half of the pulse response magnitude
                    upsampling=16, # interpolate the time domain signal to give better visulization, also allows modeling higher sampling rate without frequency domain extrapolation
                    interpolation_type='linear', # interpolation scheme, can be either 'linear' or 'cubic'
                    vh_tick = 5, # tick scale in mV for voltage axis
                    color_mask = 0.0001,
                    contour_label = False
                    ):
    '''
        https://www.oiforum.com/wp-content/uploads/2019/01/OIF-CEI-04.0.pdf
        implementation of statistical eye diagram with the inclusion of noise and jitter
        
        pulse_response: pulse response of the channel
        samples_per_symbol=128: samples per symbol
        vh_size=2048: vertical voltage discretized level
        M=4: 2 for NRZ and 4 for PAM4
        A_window_multiplier=2: control the vertical viewing space of the plot
        sample_size=16: how many symbols you want to sample across the pulse response, too many will lead to long runtime
        mu_noise=0: mean value of noise in V, assuming Gaussian
        sigma_noise=1.33e-4: std value of noise in V, assuming Gaussian
        mu_jitter=0.0125: deterministic jitter, in terms of UI
        sigma_jitter=0.015: random jitter, in terms of UI
        target_BER=2.4e-4: target BER rate
        noise_flag=False: switch for including noise or not
        jitter_flag=False: switch for including jitter or not
        plot=False: switch for plotting eye diagram or not
        pdf_conv_flag=True: if you want to do pdf convolution to find all ISI conbinations, False will then brute force find all combination
        diff_signal=True: eye diagram amplitude will be half of the pulse response magnitude
        upsampling=16: interpolate the time domain signal to give better visulization, also allows modeling higher sampling rate without frequency domain extrapolation
        interpolation_type='linear': interpolation scheme, can be either 'linear' or 'cubic'
    '''
    
    # remove the head and tail zero and remove DC component
    pulse_response = np.array(pulse_response)
    pulse_response_DC = pulse_response[0]
    pulse_response = pulse_response - pulse_response_DC # remove DC offset
    window = [i for i, e in enumerate(pulse_response) if e != 0] # window that extracts the pulse  
    window_start, window_end = window[0]-1, window[-1]+1
    if diff_signal == True:
        pulse_input = pulse_response[window_start : window_end] * 0.5  # considering differential signaling
    else:
        pulse_input = pulse_response[window_start : window_end]  # considering signal-ended signaling
        
    # oversample the signal in case there is no enough samples per symbol
    x = np.linspace(0, len(pulse_input)-1, num=len(pulse_input))
    f = interp1d(x, pulse_input, kind=interpolation_type)
    x_new = np.linspace(0, len(pulse_input)-1, num=len(pulse_input)*upsampling)
    pulse_input = f(x_new)
    samples_per_symbol = samples_per_symbol * upsampling
    window_size = samples_per_symbol
    
    idx_main = np.argmax(abs(pulse_input)) # this is the c0, main cursor, from OIF doc, see section 2.C.5 and 2.B.2

    if M == 2:
        d = np.array([-1, 1]).reshape(1,M) # direction of pulse polarity
        if pdf_conv_flag == False and sample_size >= 16:
            sample_size = 16
    elif M == 4:
        d = np.array([-1, -1/3, 1/3, 1]).reshape(1,M)  # direction of pulse polarity
        if pdf_conv_flag == False and sample_size >= 9:
            sample_size = 9
    else:
        print('M has to be either 2 or 4.')

    A_window_min = abs(pulse_input[idx_main]) * -A_window_multiplier
    A_window_max = abs(pulse_input[idx_main]) * A_window_multiplier
    mybin_edges = np.linspace(A_window_min, A_window_max, vh_size+1) # my bin edges
    vh = 0.5*(mybin_edges[1:] + mybin_edges[:-1])
    
    pdf_list = []
    for idx in range(-int(window_size/2),int(window_size/2)):
        idx_sampled = idx_main+idx
        sampled_points = []
        
        # i=0 tp include the sampled main cursor, points were sampled around the main coursor
        i = 0
        while idx_sampled - i*samples_per_symbol >= 0:
            sampled_points.append(idx_sampled - i*samples_per_symbol)
            i = i + 1
    
        # i=1 tp exclude the sampled main cursor, points were sampled around the main coursor
        j = 1 
        while idx_sampled + j*samples_per_symbol <= len(pulse_input)-1:
            sampled_points.append(idx_sampled + j*samples_per_symbol)
            j = j + 1
    
        sampled_points = sampled_points[:sample_size]
        sampled_amps = np.array([pulse_input[i] for i in sampled_points]).reshape(-1,1)
        sampled_amps = sampled_amps @ d 
        
        # using convolution to find the ISI pdf over the entire pulse response is faster than finding the all combinations then finding the histogram
        if pdf_conv_flag == True:
            pdf, _ = np.histogram(sampled_amps[0], mybin_edges) 
            pdf = pdf/sum(pdf)
            
            for j in range(1, len(sampled_amps)):
                pdf_cursor, _ = np.histogram(sampled_amps[j], mybin_edges)
                pdf_cursor = pdf_cursor/sum(pdf_cursor)
                pdf = np.convolve(pdf, pdf_cursor, mode='same')
                pdf = pdf/sum(pdf)
            
            pdf_list.append(pdf)
            
        # brute force to find all ISI combinations
        if pdf_conv_flag == False:
            all_combs = np.array(np.meshgrid(*[sampled_amps[i] for i in range(len(sampled_amps))])).T.reshape(-1,len(sampled_amps))
            A = np.sum(all_combs, axis=1)
            pdf, _ = np.histogram(A, mybin_edges)
            pdf = pdf/sum(pdf)
            pdf_list.append(pdf)
        
    ####################### noise inclusion ###########################
    hist_list = []
    if noise_flag == True:
        # mu_noise = 0
        noise_pdf = norm.pdf(vh, mu_noise, sigma_noise)
        noise_pdf = noise_pdf/sum(noise_pdf)
        for i in range(window_size):
            pdf = pdf_list[i]
            pdf = np.convolve(noise_pdf, pdf, mode='same')
            hist_list.append(pdf)
    else:
        hist_list = pdf_list
    
    ####################### jitter inclusion ###########################
    # deterministic jitter, implemented as a dual dirac function
    jitter_xaxis_step_size = 1 # typically it should be smaller than <samples_per_symbol * 0.01>
    x_axis = np.linspace(-(window_size-1), window_size-1, int(2*(window_size-1)/jitter_xaxis_step_size)+1) 
    idx_middle = int((len(x_axis)-1)/2) 
    num_steps = int(1/jitter_xaxis_step_size)
    
    # https://e2e.ti.com/blogs_/b/analogwire/posts/timing-is-everything-jitter-specifications
    mu_jitter = mu_jitter * samples_per_symbol
    sigma_jitter = sigma_jitter * samples_per_symbol
    jitter_pdf1 = norm.pdf(x_axis, -mu_jitter, sigma_jitter)
    jitter_pdf2 = norm.pdf(x_axis, mu_jitter, sigma_jitter)

    jitter_pdf = (jitter_pdf1 + jitter_pdf2) / sum(jitter_pdf1 + jitter_pdf2)
    # plt.plot(x_axis, jitter_pdf)
    
    if jitter_flag == True:
        for i in range(window_size):
            pdf = np.zeros(vh_size)
            for j in range(window_size):
                # sliding the window from index = -(idx-j) till end
                # print(idx_middle + (-(i-j)) * num_steps)
                if jitter_xaxis_step_size < 1:
                    joint_pdf = hist_list[j] * np.trapz(jitter_pdf[idx_middle+(j-i)*num_steps : idx_middle+(j-i+1)*num_steps], x_axis[idx_middle+(j-i)*num_steps : idx_middle+(j-i+1)*num_steps])  
                if jitter_xaxis_step_size == 1:
                    joint_pdf = hist_list[j] * jitter_pdf[idx_middle+(j-i)*num_steps]
                pdf = pdf + joint_pdf
    
            pdf = pdf/sum(pdf)
            hist_list[i] = pdf # overwrite the jitter included pdf

    ##################### contour ########################
    
    hist_list = np.array(hist_list).T
    # find all voltage level at eye centers
    A_pulse_max = pulse_input[idx_main]  
    if A_pulse_max >= 0:
        A_levels = A_pulse_max * d[0] * -1 # simply for consistency: we want this list to go from positive voltage levels to negative levels
    else: 
        A_levels = A_pulse_max * d[0] 
    
    if M == 4:
        if A_pulse_max >= 0: # same reason as above
            eye_center_levels = A_pulse_max * np.array([-2/3, 0, 2/3]) * -1
        else:
            eye_center_levels = A_pulse_max * np.array([-2/3, 0, 2/3])
    if M == 2:
        eye_center_levels = A_pulse_max * np.array([0])
    
    # print(eye_center_levels)
    # find all signal voltage level idx at which vertical index
    idx_A_levels_yaxis = []
    for i in range(len(A_levels)):
        idx_line_horizontal =  (np.abs(vh-A_levels[i])).argmin()
        idx_A_levels_yaxis.append(idx_line_horizontal)
        
    # find the idx of eye center on voltage axis (y-axis)
    idx_eye_center_levels_yaxis = []  # this is the idx of eye center on voltage axis
    for i in range(len(eye_center_levels)):
        idx_line_horizontal =  (np.abs(vh-eye_center_levels[i])).argmin()
        idx_eye_center_levels_yaxis.append(idx_line_horizontal)
        
    contour_list = []
    
    for i in range(window_size):
        if M == 2:
            # 1
            cdf_1 = np.cumsum(hist_list[idx_eye_center_levels_yaxis[0]:,i])
            # 0
            cdf_0 = np.cumsum(np.flip(hist_list[:idx_eye_center_levels_yaxis[0],i]))
            contour = np.concatenate((np.flip(cdf_0), cdf_1))
        if M == 4:
            # 11:
            cdf_11 = np.cumsum(hist_list[idx_eye_center_levels_yaxis[0]:,i])
            # 10:
            cdf_10_part1 = np.cumsum(np.flip(hist_list[idx_A_levels_yaxis[1]:idx_eye_center_levels_yaxis[0],i]))
            cdf_10_part2 = np.cumsum(hist_list[idx_eye_center_levels_yaxis[1]:idx_A_levels_yaxis[1],i])
            # 01:
            cdf_01_part1 = np.cumsum(np.flip(hist_list[idx_A_levels_yaxis[2]:idx_eye_center_levels_yaxis[1],i]))
            cdf_01_part2 = np.cumsum(hist_list[idx_eye_center_levels_yaxis[2]:idx_A_levels_yaxis[2],i])
            # 00:
            cdf_00 = np.cumsum(np.flip(hist_list[:idx_eye_center_levels_yaxis[2],i]))
        
            contour = np.concatenate((np.flip(cdf_00), cdf_01_part2, np.flip(cdf_01_part1), cdf_10_part2, np.flip(cdf_10_part1), cdf_11))
            # print(contour.shape)

        contour_list.append(contour)
         
    contour_list = np.array(contour_list).T
    eye = np.array(hist_list)

    ################### contour widths ####################
    try:
        idx_below_BER_horizontal_list = []
        for i in range(len(eye_center_levels)):
            contour_eye_center_horizontal = contour_list[idx_eye_center_levels_yaxis[i], :]
            idx_below_BER_horizontal = np.where(np.diff(np.signbit(contour_eye_center_horizontal-target_BER)))[0]
            idx_below_BER_horizontal_list.append(idx_below_BER_horizontal)
        
        # print(idx_eye_center_levels_yaxis)
        # print(idx_below_BER_horizontal_list)
        
        # we have to find out where the time center of the center eye first
        if M == 2:
            idx_below_BER_horizontal_center = idx_below_BER_horizontal_list[0]
        if M == 4:
            idx_below_BER_horizontal_up = idx_below_BER_horizontal_list[0]
            idx_below_BER_horizontal_center = idx_below_BER_horizontal_list[1]
            idx_below_BER_horizontal_low = idx_below_BER_horizontal_list[2]
    
        _width_center = []
        
        idx_below_BER_horizontal_center = [idx_below_BER_horizontal_center[n:n+2] for n in range(0, len(idx_below_BER_horizontal_center), 2)]
        
        for i in range(0, len(idx_below_BER_horizontal_center)):
            _ = np.diff(idx_below_BER_horizontal_center[i])[0]
            _width_center.append(_)
    
        # _width_center = np.diff(idx_below_BER_horizontal_center)
        # print(idx_below_BER_horizontal_center)

        _idx_with_center = np.argmax(_width_center)
        _idx1_width_center = idx_below_BER_horizontal_center[_idx_with_center][0]  # make an assumption here: the biggest with jump on the horizontal center line is the center eye width
        _idx2_width_center = idx_below_BER_horizontal_center[_idx_with_center][1]
        eye_width_center = (_idx2_width_center - _idx1_width_center) 
        idx_eye_center_xaxis = _idx1_width_center + int(eye_width_center/2)
        
        if M == 2:
            eye_widths = np.array([eye_width_center]) / samples_per_symbol
        if M == 4:
            _idx1_width_up = np.where(np.diff(np.signbit(idx_below_BER_horizontal_up-idx_eye_center_xaxis)))[0][0]
            _idx2_width_up = _idx1_width_up + 1
            eye_width_up = idx_below_BER_horizontal_up[_idx2_width_up] - idx_below_BER_horizontal_up[_idx1_width_up]
            
            _idx1_width_low = np.where(np.diff(np.signbit(idx_below_BER_horizontal_low-idx_eye_center_xaxis)))[0][0]
            _idx2_width_low = _idx1_width_low + 1
            eye_width_low = idx_below_BER_horizontal_low[_idx2_width_low] - idx_below_BER_horizontal_low[_idx1_width_low]
    
            eye_widths = np.array([eye_width_up, eye_width_center, eye_width_low])/samples_per_symbol
    
        eye_widths_mean = np.mean(eye_widths)
        # print(idx_eye_center_xaxis)
        # print(eye_widths)
    except:
        eye_widths_mean = 0
        if M == 4:
            eye_widths = [0, 0, 0]
        if M == 2:
            eye_widths = 0
            
    ################ contour heights, distortion heights and COM #################
    
    try:
        contour_eye_center_vertical = contour_list[:, idx_eye_center_xaxis]
        idx_below_BER_vertical = np.where(np.diff(np.signbit(contour_eye_center_vertical-target_BER)))[0]
        # print(idx_below_BER_vertical)
        
        eye_heights = []
        for j in range(0, 2*(M-1), 2): # it finds the BER contour heights from the vertical line
            _idx1_eye_height = idx_below_BER_vertical[j]
            _idx2_eye_height = idx_below_BER_vertical[j+1]
            eye_height = vh[_idx2_eye_height] - vh[_idx1_eye_height]
            eye_heights.append(eye_height)
    
        eye_heights = np.flip(eye_heights) # so that it is from up eye to bottom eye
        eye_heights_mean = np.mean(eye_heights)
    
        # distortion heights #
    
        idx_distortion = idx_below_BER_vertical.copy()
        idx_distortion = np.insert(idx_distortion, 0, idx_A_levels_yaxis[-1])
        idx_distortion = np.append(idx_distortion, idx_A_levels_yaxis[0])
        
        distortion_heights = []
        for i in range(0, 2*M, 2):
            idx1_distortion = idx_distortion[i] 
            idx2_distortion = idx_distortion[i+1] 
            distortion_height = vh[idx2_distortion] - vh[idx1_distortion]
            distortion_heights.append(distortion_height)
    
        distortion_heights = np.flip(distortion_heights) # so that it is from up eye to bottom eye
        distortion_heights_mean = np.mean(distortion_heights)
        
        # COM #
        COM = 20*np.log10((A_levels[0]-A_levels[-1])/np.sum(distortion_heights))

    except:
        COM = 0   
        distortion_heights = A_levels
        distortion_heights_mean = np.mean(distortion_heights)
        if M == 4:
            eye_heights = [0, 0, 0]
        if M == 2:
            eye_heights = 0
        eye_heights_mean = 0

    ##################### plot eye diagram ##########################
    # show heat map of the eye diagram
    # https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    # plot = True
    if plot == True:
        fig, ax = plt.subplots(1,1)
        
        _y_axis_heatmap_up = np.linspace(0, A_window_max, int(vh_size/2)+1)[1:] * 1e3 # mV
        _y_axis_heatmap_down = np.linspace(A_window_min, 0, int(vh_size/2)+1)[1:] * 1e3 #mV
        y_axis_heatmap = np.concatenate((_y_axis_heatmap_down, _y_axis_heatmap_up))
        yticklabels = np.flip(y_axis_heatmap)
        _tick_up = []
        i = 0
        while i * vh_tick <= A_window_max * 1e3:
            _tick_up.append(i)
            i = i + 1
            
        _tick_up = np.array(_tick_up) * vh_tick
        _tick_down = _tick_up * -1
        _tick = np.sort(np.concatenate((_tick_up, _tick_down)))
            
        for i in _tick:
            idx = (np.abs(yticklabels-i)).argmin()
            yticklabels[idx] = i
                    
        _tick_time = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
        xticklabels = np.arange(int(-window_size/2), int(window_size/2))/samples_per_symbol
        for i in _tick_time:
            idx = (np.abs(xticklabels-i)).argmin()
            xticklabels[idx] = i
        
        eye_df = pd.DataFrame(eye)
        # color bar notation in scientific notation
        fmt = ticker.ScalarFormatter(useMathText=(True))
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2,2))
        heatmap = sns.heatmap(data=eye_df, xticklabels=xticklabels, yticklabels=yticklabels, cmap='rainbow', mask=(eye_df<=color_mask), cbar_kws={'format': fmt})
              
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=14)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=14)
        
        # reduce the density of x axis
        for ind, label in enumerate(heatmap.get_xticklabels()):
            if float(label.get_text())*10 % 1 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
                ax.xaxis.get_major_ticks()[ind].tick1line.set_visible(False)
        
        # reduce the density of y axis
        for ind, label in enumerate(heatmap.get_yticklabels()): 
            if float(label.get_text()) % 5 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
                ax.yaxis.get_major_ticks()[ind].tick1line.set_visible(False)
        
        # add frame to the heatmap
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
        
        contour_plot = ax.contour(np.arange(0.5, window_size), np.arange(0.5, vh.shape[0]), np.array(contour_list), levels=[target_BER], colors='black')
        if contour_label == True:
            ax.clabel(contour_plot, inline=True, fmt='%.1e')
        if noise_flag == True and jitter_flag == False:
            ax.set_title('$\mu_{{noise}}$={:.2e} samples | $\sigma_{{noise}}$={:.2e} V \n'.format(mu_noise, sigma_noise))
        elif jitter_flag == True and noise_flag == False:
            ax.set_title('$\mu_{{jitter}}$={:.2e} UI | $\sigma_{{jitter}}$={:.2e} UI \n'.format(mu_jitter/samples_per_symbol, sigma_jitter/samples_per_symbol))
        elif jitter_flag == True and noise_flag == True:
            ax.set_title('''$\mu_{{noise}}$={:.2e} V | $\sigma_{{noise}}$={:.2e} V 
                                 $\mu_{{jitter}}$={:.2e} UI | $\sigma_{{jitter}}$={:.2e} UI \n'''.format(mu_noise, sigma_noise, mu_jitter/samples_per_symbol, sigma_jitter/samples_per_symbol))
        else:
            ax.set_title('Statistical Eye without Jitter or Noise')
        
        ax.set_ylabel('Voltage (mV)', fontsize=14)
        ax.set_xlabel('Time (UI)', fontsize=14)
        # save the picture to disk
        fig.savefig(f'pics/stateye_{time_string}.eps', format='eps', bbox_inches='tight')
        fig.savefig(f'pics/stateye_{time_string}.png', format='png', bbox_inches='tight')
        
    return{'center_COM (dB)': COM,
           'eye_heights (V)': eye_heights,
           'eye_heights_mean (V)': eye_heights_mean,
           'distortion_heights (V)': distortion_heights,
           'distortion_heights_mean (V)': distortion_heights_mean,
           'eye_widths (UI)': eye_widths,
           'eye_widths_mean (UI)': eye_widths_mean,
           'A_levels (V)': A_levels,
           'eye_center_levels (V)': eye_center_levels,
           'stateye': eye,
        }

if __name__ == "__main__":

    pulse_response_dir = '/autofs/fs1.ece/fs1.eecg.tcc/lizongh2/serdes/pulse_response/'
    pulse_response_name = 'ml_final_250um.csv'
    # noise:
        # 250um: 7.384949841885791e-04
        # 500um: 7.413938529920631e-04
        # 5mm: 7.411989548792984e-04
    channel_pulse_response = pd.read_csv(pulse_response_dir+pulse_response_name).to_numpy().reshape(-1)
    idx_main = np.argmax(abs(channel_pulse_response)) # this is the c0, main cursor, from OIF doc, see section 2.C.5 and 2.B.2

    _ = statistical_eye(pulse_response=channel_pulse_response, 
                                            samples_per_symbol=8, 
                                            A_window_multiplier=2, 
                                            sigma_noise=7.384949841885791e-04, 
                                            M=4, 
                                            sample_size=32, 
                                            target_BER=2.4e-4,
                                            plot=True, 
                                            noise_flag=True, 
                                            jitter_flag=True,
                                            mu_jitter=0,
                                            sigma_jitter=1.5e-2)

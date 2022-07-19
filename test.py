import numpy as np
from statistical_eye import statistical_eye

with open('channel_pulse_response_test.csv') as file_name:
    pulse_response = np.loadtxt(file_name, delimiter=",")

results = statistical_eye(pulse_response=pulse_response, 
                                        idx_main = np.argmax(abs(pulse_response))
                                        samples_per_symbol=128, 
                                        A_window_multiplier=1.5, 
                                        M=4, 
                                        sample_size=16, 
                                        target_BER=2.4e-4,
                                        plot=True, 
                                        noise_flag=False, 
                                        jitter_flag=False,
                                        upsampling=1)

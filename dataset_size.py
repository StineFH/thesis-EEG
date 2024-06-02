import mne
import mne_bids as mb
import data_utils4 as du

# bidsPath = 'Y:\\NTdata\\BIDS\\EESM19\\derivatives\\cleaned_1\\'
bidsPath = '/data/'
subjectIds=mb.get_entity_vals(bidsPath,'subject', with_key=False)
paths=du.returnFilePaths(bidsPath,subjectIds[6:],sessionIds=['001', '002', '003', '004'])

n_times = []
seconds = []

for f in paths:
    tempRaw=mne.io.read_raw_eeglab(f, preload=True,verbose=False)
    n_times.append(tempRaw.n_times)
    seconds.append(tempRaw.times[-1])
    print(tempRaw)

print("******************** DONE ********************************")

print(f'smallet series {min(n_times)} and largest {max(n_times)}')
print(f'Data points in total {sum(n_times)}')
print(f'smallet series by seconds {min(seconds)} and longest {max(seconds)}')
print(f'Seconds in total {sum(seconds)}')
print(f'This make {sum(seconds)/60} minutes and {(sum(seconds)/60)/60} hours')


# 413433020*25
# 10,335,825,500
# # For one channel 
# 459.36996 *3
# 413433020*3
# (413433020*3)*0.7
# (1240299060/250/60)/60

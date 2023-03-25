from scipy.signal import find_peaks
import numpy as np

def subseq(arr, n):
    i, inc_count, dec_count = 0, 0, 0
    max = [0]*n
    min = [0]*n

    k1 = 0
    k2 = 0

    if (arr[0] < arr[1]):
        min[k1] = 0
        k1 += 1
    else:
        max[k2] = 0
        k2 += 1

    for i in range(1, n-1):
        if (arr[i] < arr[i - 1] and arr[i] < arr[i + 1]):
            min[k1] = i
            k1 += 1
 
        if (arr[i] > arr[i - 1] and arr[i] > arr[i + 1]):
            max[k2] = i
            k2 += 1

    if (arr[n - 1] < arr[n - 2]):
        min[k1] = n - 1
        k1 += 1
    else:
        max[k2] = n - 1
        k2 += 1

    if (min[0] == 0):
        inc_count = k2
        dec_count = k1 - 1
    else:
        inc_count = k2 - 1
        dec_count = k1
    
    return inc_count + dec_count

def peaks(arr):
    return find_peaks(arr, prominence = 20, threshold=8)

def binary(arr):
    old_state = None
    changes = []
    for i in range(len(arr)):
        state = 'high' if arr[i] == 255 else 'low'

        if not old_state or old_state != state:
            changes.append((i, state))
            old_state = state
        
    return dict(changes = changes)

def moving_window_averages(arr, window=5, thres=8.0):
    changes = []
    rolling = [None] * window
    old_state = None

    rolling[window-1] = 0

    for i in range(window, len(arr) - 1):
        slc = arr[i - window:i + 1]
        mean = sum(slc) / float(len(slc))
        state = 'good' if mean > (rolling[i-1] + thres) else 'bad'

        rolling.append(mean)
        if not old_state or old_state != state:
            #print('Changed to {:>4s} at position {:>3d} ({:5.3f})'.format(state, i, mean))
            changes.append((i, state))
            old_state = state

    return dict(arr = arr,
                rolling = rolling,
                thres = thres,
                changes = changes)
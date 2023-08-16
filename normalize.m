function N = normalize(data)
N = (data - min(data)) / ( max(data) - min(data) );
import numpy as np
import json



import json
import numpy as np

def normalize():
    # Load data from JSON file
    with open('data/uptown_funk.json') as f:
        data = json.load(f)
    
    # Find the max and min values for x and y coordinates
    x_max, y_max = -np.inf, -np.inf
    x_min, y_min = np.inf, np.inf
    for key, values in data.items():
        for x, y in values:
            x_max = max(x_max, x)
            x_min = min(x_min, x)
            y_max = max(y_max, y)
            y_min = min(y_min, y)
    
    # Normalize the data using min-max scaling
    data_normalized = {}
    for key, values in data.items():
        values_normalized = []
        for x, y in values:
            x_norm = (x - x_min) / (x_max - x_min) - 0.5
            y_norm = (y - y_min) / (y_max - y_min) - 0.5
            values_normalized.append([x_norm, y_norm])
        data_normalized[key] = values_normalized
    
    # Write the normalized data to a new JSON file
    with open('data/normalized_uptown_funk.json', 'w') as f:
        json.dump(data_normalized, f)



if __name__=='__main__':
    normalize()
# Imports
import csv
import numpy as np

def extract_data_from_csv(filename):
    """
    Extracts data from a CSV file with the given format.

    Args:
        filename: Path to the CSV file.

    Returns:
        A dictionary where keys are column names and values are lists of data.
    """

    data = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        header = next(reader)  # Read the header row
        for row in reader:
            for i, key in enumerate(header):
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(row[i]))
                except ValueError:
                    data[key].append(row[i])  # Handle non-numeric values

    size = 0
    for key, val in data.items():
        if size and size != len(val):
            raise ValueError("Bad CSV")
        size = len(val)

    return data, size

class Frenet:
    def __init__(self, x, y, filename):
        self.racing_line, self.size = extract_data_from_csv(filename)
        self.window_size = 10 # Find the optimal window size
        self.waypoint_idx = 0
        self.waypoint_idx = self.find_closest_point(x, y, optimize=False)

    def get_racing_line(self):
        return self.racing_line
    
    def find_closest_point(self, x, y, optimize):
        """
        Finds the closest point to the given absolute coordinates.
        """
        start_idx = 0
        end_idx = self.size
        if optimize:
            start_idx = self.waypoint_idx
            end_idx = (start_idx + self.window_size) % self.size

        idx = start_idx
        dist = float('inf')
        while True:
            point = [self.racing_line[" x_m"][idx], self.racing_line[" y_m"][idx]]
            distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if distance <= dist:
                self.waypoint_idx = idx

            idx += 1
            if idx == end_idx:
                break
            idx %= self.size
        
        print(self.waypoint_idx)
        return self.waypoint_idx

filename = "/F1tenth_AutoPursuit/maps/BrandsHatch/BrandsHatch_raceline.csv"
frenet = Frenet(0, 0, filename)
frenet.find_closest_point(5, 3, optimize=True)
# Secondary.py
import numpy as np

class Secondary():
    # Just treat this as a Schwarzschild Black Hole
    def __init__(self, mass):
        self.mass = mass
        self.radius = 2 * self.mass

    def cross_section(self):
        return 4 * np.pi * (self.radius)**2

class KerrHole(Secondary):
    def __init__(self, spin, orientation=None):
        super().__init__()
        self.spin = spin
        self.radius = 1 + np.sqrt(1 - spin**2)
        if orientation is None:
            self.orientation = np.array([0.0, 0.0, 1.0])
        else:
            self.orientation = np.array(orientation)/np.linalg.norm(orientation)

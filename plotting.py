import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

class Slice:
    '''
    class untuk buat objek slicing peta.
    '''
    
    def __init__(self, t1, t2, df, s=None):
        self.t1, self.t2 = t1, t2
        self.x, self.y, self.z = df['Easting'] , df['Northing'],  df['Depth']
        self.s = s
        
        if s:
            self._create_rectangle()
            self._inside_bound()
            self._project_to_slice()
            
        else:
            self._project_to_slice()
            
    
    def _create_rectangle(self):
        # create new basis vector
        b1 = self.t2 - self.t1
        rot = np.array([[0, 1], 
                        [-1, 0]])
        b2 = rot.dot(b1)
        new_basis = np.array([[b2[0], b1[0]],
                              [b2[1], b1[1]]])

        # vector length of the desired window in new basis vector 
        n = self.s / np.linalg.norm(b2)
        v = new_basis.dot(np.array([[n], [0]]))
        ds = v[:, 0]
    
        # rectangle vertices
        # titik akhir harus = titik awal
        # agar membentuk poligon tertutup
        rectangle = np.array([[self.t1 + ds],
                              [self.t2 + ds],
                              [self.t2 - ds],
                              [self.t1 - ds],
                              [self.t1 + ds]])
        
        self.bound = rectangle[:, 0]
        return self
    
    def _inside_bound(self):
        p = Path(self.bound)
        mask = p.contains_points(np.c_[self.x, self.y])
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.z = self.z[mask]
        return self
    
    def _project_to_slice(self):
        '''
        p1 = coordinate of 1 point
        p2 = coordinate of another point
        p1 and p2 forms a line in which we want to project
        points with x and y coordinates into

        returns distance of point (x,y) relative to point p1
        '''
        v = (self.t2 - self.t1).reshape(1, 2)
        xy = np.c_[self.x, self.y].T
        xy = xy - self.t1[:, None]
        newx = (v @ xy) / np.linalg.norm(v)
        self.newx = newx
        self.line_length = np.linalg.norm(v)
        return self
    
    def plot_map(self, ax=None, **params):
        if ax is None:
            ax = plt.gca()
        
        if self.s is not None:
            p = Path(self.bound)
            patch = patches.PathPatch(p, facecolor='orange', lw=2, alpha=0.5)   
            ax.add_patch(patch)
        
        ax.scatter(self.x, self.y, c='r', **params)
        ax.plot((self.t1[0], self.t2[0]), 
                (self.t1[1], self.t2[1]), 'ro-')
        
        ax.annotate("A", 
                    xy = (self.t1[0], self.t1[1]),
                    size=15)
        ax.annotate("A'", 
                    xy = (self.t2[0], self.t2[1]),
                    size=15)
        
        ax.set_aspect('equal')
        
    def plot_vsection(self, ax=None, **params):
        if ax is None:
            ax = plt.gca()
        
        ax.scatter(self.newx, self.z, **params)
        ax.annotate("A", 
                    xy = (np.min(self.newx), 
                          np.min(self.z)),
                    xytext=(-0.2, 1), 
                    size=15
                   )
        
        ax.annotate("A'", 
                    xy=(np.max(self.newx), 
                        np.min(self.z)),
                    xytext=(self.line_length, 1), 
                    size=15
                   )
        
        ax.set_xlim(0, self.line_length)
        ax.invert_yaxis()
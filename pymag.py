import cupy as np
import math
import copy
import itertools
import sys


"""
Default valeus
Ms = 1000 # [unit : emu / cc]
gamma = 1.76*10**7 # [unit : Oe^-1 s^-1]
alpha = 0.1 # [unit : dimensionless]
Ku = 8.0*10**6 # [unit : erg / cc]
Aex = 1.5*10**-6 # [unit : erg / cm]
DDMI = 3.5 # [unit : erg / cm^2]
zetaDLT = -0.4 # [unit : dimensionless]
zetaFLT = 0.0 # [unit : dimensionless]
Hextx, Hexty, Hextz = 0, 0, 0 # [unit : Oe]
T = 0 # [unit : K]
"""

#############################################################################################################################################
# physical constants
#############################################################################################################################################

hbar = 1.05457266*10**-27  # [unit : erg s]
echarge = 1.60217662*10-19 # [unit : A s]
kB = 1.380648*10**-16  # [unit : erg / K]
degree = math.pi/180

#############################################################################################################################################
# utils
#############################################################################################################################################

def MatrixCrossProduct(Mat1, Mat2):
    """
    Returns the cross products of Mat1 and Mat2.
    :param:
        - Mat1 & Mat2 - Required  :  5D matrix with shape (3,1,nz,ny,nx).
    :return: 
        - Mat3                    :  5D matrix with shape (3,1,nz,ny,nx).
    """
    Mat3 = np.zeros_like(Mat1)
    Mat3[0] = Mat1[1]*Mat2[2]-Mat1[2]*Mat2[1]
    Mat3[1] = Mat1[2]*Mat2[0]-Mat1[0]*Mat2[2]
    Mat3[2] = Mat1[0]*Mat2[1]-Mat1[1]*Mat2[0]
    return Mat3

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    :params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    output = '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    sys.stdout.write(output + '\n')
#     # Print New Line on Complete
#     if iteration == total:
#         print(output, end = '\n')

    
def are_space_the_same(obj1, obj2):
    return (obj1.Lx == obj2.Lx) & \
           (obj1.Ly == obj2.Ly) & \
           (obj1.Lz == obj2.Lz) & \
           (obj1.nx == obj2.nx) & \
           (obj1.ny == obj2.ny) & \
           (obj1.nz == obj2.nz)

def are_overlapped(obj1, obj2):
    f1 = obj1.clone()
    f2 = obj2.clone()
    intersection = f1.mask * f2.mask
    area = intersection.sum().tolist()
    return (area>0)

def can_concatenated(obj1, obj2, axis=0):
    f1 = obj1.clone()
    f2 = obj2.clone()
    are_the_same_spec_length = (round(f1.lcx, 16) == round(f2.lcx, 16)) & \
                               (round(f1.lcy, 16) == round(f2.lcy, 16)) & \
                               (round(f1.lcz, 16) == round(f2.lcz, 16))
    if are_the_same_spec_length == False:
        raise ValueError("The specific lenght of the two space are not the same.")
    if axis==0:
        return (f1.nx == f2.nx) & (f1.ny == f2.ny)
    if axis==1:
        return (f1.nz == f2.nz) & (f1.nx == f2.nx)
    if axis==2:
        return (f1.ny == f2.ny) & (f1.nz == f2.nz)

# 디렉토리가 있는지 확인하고 없다면 생성하는 함수
import os
def make_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        

import csv
class MagnetLogger():
    """
    mlogger = MagnetLogger('new.csv')
    
    mlogger(1,2,3,4,5)
    
    """
    def __init__(self, log_name):
        make_path('./log')
        self.path = './log/'+log_name+'.csv'
        
        if os.path.isdir(self.path):
            os.remove(self.path)
        
        self.__call__('time (s)', 'dm', 'mean_mx', 'mean_my', 'mean_mz')
    
    def __call__(self, time=None, dm=None, mean_mx=None, mean_my=None, mean_mz=None):
        f = open(self.path, 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([time, dm, mean_mx, mean_my, mean_mz])
        f.close()
        
    def load_log(self):
        f = open(self.path, 'r')
        rdr = csv.reader(f)
        
        times    = []
        dms      = []
        mean_mxs = []
        mean_mys = [] 
        mean_mzs = []
        for line in rdr:
            time_log, dm_log, mean_mx_log, mean_my_log, mean_mz_log = line
            if dm_log!='dm':
                time_log = float(time_log)
                dm_log = float(dm_log)
                mean_mx_log = float(mean_mx_log)
                mean_my_log = float(mean_my_log)
                mean_mz_log = float(mean_mz_log)
                times.append(time_log)
                dms.append(dm_log)
                mean_mxs.append(mean_mx_log)
                mean_mys.append(mean_my_log)
                mean_mzs.append(mean_mz_log)
            
        f.close()
        return times, dms, mean_mxs, mean_mys, mean_mzs
    

import pickle

def save_Evolver(evolver, evolver_name):
    make_path('./evolver')
    save_dir = './evolver/'+evolver_name+'.pkl'
    print('{} is saved at {}'.format(evolver_name, save_dir))
    with open(save_dir, 'wb') as f:
        pickle.dump(evolver, f)
        
def load_Evolver(evolver_name):
    load_dir = './evolver/'+evolver_name+'.pkl'
    print('{} is loaded from {}'.format(evolver_name, load_dir))
    with open(load_dir, 'rb') as f:
        evolver = pickle.load(f)
    return evolver

def is_conversion(x, criteria=0.1, min_iter=10):
    """
    Return the 1d array is conversion or not.
    :params:
        x              - Required  : array (Array)
        criteria       - Optional  : conversion error criteria (float)
        min_iter       - Optional  : receptive range (Int)
    """
    if len(x.shape)!=1:
        raise ValueError('x should be 1d.')
    
    i = len(x) - 1
    
    if i > min_iter:
        middle = min_iter//2
        a = sum(x[i+1-min_iter:i-middle])
        b = sum(x[i+1-middle:i])
        conv_error = np.abs(a - b) / np.abs(b)
    else:
        conv_error = 100
    
    print('Conversion Error : {}'.format(conv_error))
    
    if conv_error<criteria:
        res = True
    else:
        res = False
    return res

#############################################################################################################################################
# magmetic object
#############################################################################################################################################
  
class MagneticObject():
    """
    Magnetic object class.
    """
    def __init__(self, 
                 Lx=100*10**-7, Ly=100*10**-7, Lz=5*10**-7, 
                 nx=100, ny=100, nz=5, 
                 PBCx=False, PBCy=False, PBCz=False):
        """
        Initialize a magnetic object.
        :params:
            Lx, Ly, Ly          - Required  : length of the magnetic object along x,y,z axis [unit: cm] (float)
            nx, ny, nz          - Required  : number of discrete elements along x,y,z axis (int)
            PBCx, PBCy, PBCz    - Optional  : Periodic boundary condition (bool)
        """
        
        # Real size factor
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        
        # Number of cells
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Cherecteristic length
        self.lcx = self.Lx / self.nx
        self.lcy = self.Ly / self.ny
        self.lcz = self.Lz / self.nz

        # Periodic boundary condition
        self.PBCx = PBCx
        self.PBCy = PBCy
        self.PBCz = PBCz
        
        # Mask
        self.mask = np.zeros(shape=(self.nz, self.ny, self.nx))
        
    def generateMask(self):
        """
        Seperate self.mask to self.mask_face and self.mask_edge.
        :properties:
            self.mask      : mask of the magnetic object
            self.mask_zp   : -z side edge compoment of the mask
            self.mask_zm   : +z side edge component of the mask
            self.mask_yp   : -y side edge component of the mask
            self.mask_ym   : +y side edge component of the mask
            self.mask_xp   : -x side edge component of the mask
            self.mask_xm   : +x side edge component of the mask
            self.mask_edge : total edge component of the mask
            self.mask_face : face component of the mask
        """
        self.mask_zp = np.roll(self.mask, shift=+1, axis=0)
        self.mask_zm = np.roll(self.mask, shift=-1, axis=0)
        self.mask_yp = np.roll(self.mask, shift=+1, axis=1)
        self.mask_ym = np.roll(self.mask, shift=-1, axis=1)
        self.mask_xp = np.roll(self.mask, shift=+1, axis=2)
        self.mask_xm = np.roll(self.mask, shift=-1, axis=2)
        
        if self.PBCz==True:
            self.mask_zp = (self.mask - self.mask_zp)*self.mask
            self.mask_zm = (self.mask - self.mask_zm)*self.mask
        elif self.PBCz==False:
            mask_ = np.pad(self.mask, ((0,1),(0,0),(0,0)), 'constant', 
                           constant_values=((0,0),(0,0),(0,0)))
            mask_zp_ = np.roll(mask_, shift=+1, axis=0)
            mask_zm_ = np.roll(mask_, shift=-1, axis=0)
            mask_zp_ = (mask_ - mask_zp_)*mask_
            mask_zm_ = (mask_ - mask_zm_)*mask_
            self.mask_zp = mask_zp_[:-1,:,:]
            self.mask_zm = mask_zm_[:-1,:,:]
        
        if self.PBCy==True:
            self.mask_yp = (self.mask - self.mask_yp)*self.mask
            self.mask_ym = (self.mask - self.mask_ym)*self.mask
        elif self.PBCy==False:
            mask_ = np.pad(self.mask, ((0,0),(0,1),(0,0)), 'constant', 
                           constant_values=((0,0),(0,0),(0,0)))
            mask_yp_ = np.roll(mask_, shift=+1, axis=1)
            mask_ym_ = np.roll(mask_, shift=-1, axis=1)
            mask_yp_ = (mask_ - mask_yp_)*mask_
            mask_ym_ = (mask_ - mask_ym_)*mask_
            self.mask_yp = mask_yp_[:,:-1,:]
            self.mask_ym = mask_ym_[:,:-1,:]
        
        if self.PBCx==True:
            self.mask_xp = (self.mask - self.mask_xp)*self.mask
            self.mask_xm = (self.mask - self.mask_xm)*self.mask
        elif self.PBCx==False:
            mask_ = np.pad(self.mask, ((0,0),(0,0),(0,1)), 'constant', 
                           constant_values=((0,0),(0,0),(0,0)))
            mask_xp_ = np.roll(mask_, shift=+1, axis=2)
            mask_xm_ = np.roll(mask_, shift=-1, axis=2)
            mask_xp_ = (mask_ - mask_xp_)*mask_
            mask_xm_ = (mask_ - mask_xm_)*mask_
            self.mask_xp = mask_xp_[:,:,:-1]
            self.mask_xm = mask_xm_[:,:,:-1]
        
        self.mask_zp = self.mask_zp.astype(bool)
        self.mask_zm = self.mask_zm.astype(bool)
        self.mask_yp = self.mask_yp.astype(bool)
        self.mask_ym = self.mask_ym.astype(bool)
        self.mask_xp = self.mask_xp.astype(bool)
        self.mask_xm = self.mask_xm.astype(bool)
        
        self.mask_edge = self.mask_zp | self.mask_zm | self.mask_yp | self.mask_ym | self.mask_xp | self.mask_xm
        self.mask_face = self.mask.astype(bool) & ~self.mask_edge
        
    def clone(self):
        """
        Clone the magnetic object.
        """
        return copy.deepcopy(self)

    def intersection(self, obj):
        """
        Intersect another object, which overlap with the object, along an axis.
        The intersection must be done before properties of the two objects is defined. 
        :params:
            obj      - Required  : another object.
        """
        if are_space_the_same(self, obj)==False:
            raise ValueError("The space parameters of the two object are not the same.")
        f1 = self.clone()
        f2 = obj.clone()
        intersection = f1.mask * f2.mask
        f1.mask = intersection
        f1.generateMask()
        return f1
        
    def union(self, obj):
        """
        Union another object, which overlap with the object, along an axis.
        The union must be done before properties of the two objects is defined. 
        :params:
            obj      - Required  : another object.
        """
        if are_space_the_same(self, obj)==False:
            raise ValueError("The space parameters of the two object are not the same.")
        f1 = self.clone()
        f2 = obj.clone()
        intersection = f1.mask * f2.mask
        union = f1.mask + f2.mask - intersection
        f1.mask = union
        f1.generateMask()
        return f1
    
    def difference(self, obj):
        """
        Substract another object from the object.
        The difference must be done before properties of the two objects is defined. 
        :params:
            obj      - Required  : another object.
        """
        if are_space_the_same(self, obj)==False:
            raise ValueError("The space parameters of the two object are not the same.")
        f1 = self.clone()
        f2 = obj.clone()
        intersection = f1.mask * f2.mask
        f1.mask = f1.mask - intersection
        f1.generateMask()
        return f1
    
    def asign_material(self, mater):
        """
        Asign materials to the mask.
        :params:
            mater    - Required   : instance of PycroMagnetics.magmatlib.
        """
        self.Ms = mater.Ms
        self.Aex = mater.Aex
        self.gamma = mater.gamma
        self.alpha = mater.alpha
        self.Ku = mater.Ku
        self.thetaK = mater.thetaK
        self.phiK = mater.phiK
        self.K1 = mater.K1
        self.K2 = mater.K2
        self.DDMI = mater.DDMI
        self.Temp = mater.Temp

    def setCylindricalMask(self, 
                           center_x=50*10**-7, center_y=50*10**-7, center_z=2.5*10**-7, 
                           radius_x=40*10**-7, radius_y=40*10**-7, 
                           angle=0, height=4*10**-7):
        """
        Generate a magnetic object with ellipsoidal disk shape.
        :params:
            center_x, center_y, center_z   - Required  : central position of the mask [unit: cm] (float)
            radius_x                       - Required  : first axis radius [unit: cm] (float)
            radius_y                       - Required  : second axis radius [unit: cm] (float)
            height                         - Required  : height of the mask [unit: cm] (float)
            angle                          - Optional  : rotation angle [unit: deg] (float)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.height = height
        self.angle = angle

        for k in range(self.nz):
            z = (k + 0.5) * self.lcz
            z = round(z, 16)
            cart = np.abs(z - self.center_z)
            if cart <= self.height/2:
                Y, X = np.ogrid[:self.ny, :self.nx]
                X = (X + 0.5) * self.lcx - self.center_x
                Y = (Y + 0.5) * self.lcy - self.center_y
                X = X.round(decimals=16)
                Y = Y.round(decimals=16)
                X2 = X*np.cos(self.angle*math.pi/180) + Y*np.sin(self.angle*math.pi/180) 
                Y2 = -X*np.sin(self.angle*math.pi/180) + Y*np.cos(self.angle*math.pi/180) 
                X,Y = X2/radius_x,Y2/radius_y
                dist_from_center = np.sqrt(X**2 + Y**2)
                self.mask[k] = dist_from_center < 1
                
        self.generateMask()

#############################################################################################################################################
# Ferromatnet
#############################################################################################################################################

class Ferromagnet(MagneticObject):
    """
    Ferromagnet class.
    """
    def __init__(self, **kwargs):
        """
        Initialize a ferromagnet object.
        In this stage, the mask of the ferromagnet is not defined.
        In order to generate the mask of the ferromagnet, you should use a mask generating method, for example set setCylindricalMask
        :example:
            # define a ferromaget instance
            m1 = Ferromagnet(Lx=50*10**-7, 
                             Ly=50*10**-7, 
                             Lz=50*10**-7, 
                             nx=50, 
                             ny=50, 
                             nz=50,
                             PBCx=False,
                             PBCy=False,
                             PBCz=False)

            # define a mask of the instance
            m1.setCylindricalMask(center_x=10*10**-7, 
                                  center_y=10*10**-7, 
                                  center_z=25*10**-7, 
                                  radius_x=4*10**-7, 
                                  radius_y=4*10**-7, 
                                  height=10*10**-7,
                                  angle=0)

            # define magnetic properties
            m1.Ms = 1080 # [unit : emu / cc]
            m1.Aex = 1.5*10**-6 # [unit : erg / cm]
            m1.gamma = 1.76*10**7 # [unit : Oe^-1 s^-1]
            m1.alpha = 10 # [unit : dimensionless]
            m1.Ku = 8.0*10**6 # [unit : erg / cc]
            m1.thetaK = 0 # [unit : deg]
            m1.phiK = 0 # [unit : deg]
            m1.DDMI = 3.5 # [unit : erg / cm^2]
            m1.Temp = 300 # [unit : K]

            # set magnetization direction
            m1.setUniformMagnetization(thetaM0=0, phiM0=0)
        """
        super(Ferromagnet, self).__init__(**kwargs)
        self.Ms = 0 
        self.Aex = 0
        self.gamma = 0
        self.alpha = 0
        self.Ku = 0
        self.thetaK = 0
        self.phiK = 0
        self.K1 = 0
        self.K2 = 0
        self.DDMI = 0
        self.Temp = 0

    # Saturation magnetization
    @property
    def Ms(self):
        return self.__Ms

    @Ms.setter
    def Ms(self, value):
        """
        Set a property `saturation magnetization` [unit: emu/cc].
        The the property is only applied to the mask of the instance
        """
        self.__Ms = value*self.mask

    # Exchagne stiffness
    @property
    def Aex(self):
        return self.__Aex

    @Aex.setter
    def Aex(self, value):
        """
        Set a property `exchange stiffnet` [unit: erg/cc].
        The the property is only applied to the mask of the instance
        """
        self.__Aex = value*self.mask

    # Gyromagnetic ratio
    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        """
        Set a property `gyromagnetic ratio` [unit : Oe^-1 s^-1].
        The the property is only applied to the mask of the instance
        """
        self.__gamma = value*self.mask

    # Damping constant
    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        """
        Set a property `damping constant` [unit : dimensionless].
        The the property is only applied to the mask of the instance
        """
        self.__alpha = value*self.mask

    # DMI
    @property
    def DDMI(self):
        return self.__DDMI

    @DDMI.setter
    def DDMI(self, value):
        """
        Set a property `DMI constant` [unit : erg / cm^2].
        The the property is only applied to the mask of the instance
        """
        self.__DDMI = value*self.mask

    # Uniaxial anisotropy
    @property
    def Ku(self):
        return self.__Ku

    @Ku.setter
    def Ku(self, value):
        """
        Set a property `uniaxial anisotropy energy density` [unit : erg / cc].
        The the property is only applied to the mask of the instance
        """
        self.__Ku = value*self.mask

    @property
    def thetaK(self):
        return self.__thetaK

    @thetaK.setter
    def thetaK(self, value):
        """
        Set a property `polar angle of Ku axis` [unit : deg].
        The the property is only applied to the mask of the instance
        """
        self.__thetaK = value*self.mask

    @property
    def phiK(self):
        return self.__phiK

    @phiK.setter
    def phiK(self, value):
        """
        Set a property `azimuth angle of Ku axis` [unit : deg].
        The the property is only applied to the mask of the instance
        """
        self.__phiK = value*self.mask

        
    # Uniaxial anisotropy
    @property
    def K1(self):
        return self.__K1

    @K1.setter
    def K1(self, value):
        """
        Set a property `first cubic anisotropy energy density` [unit : erg / cc].
        The the property is only applied to the mask of the instance
        """
        self.__K1 = value*self.mask
        
    @property
    def K2(self):
        return self.__K2

    @K2.setter
    def K2(self, value):
        """
        Set a property `second cubic anisotropy energy density` [unit : erg / cc].
        The the property is only applied to the mask of the instance
        """
        self.__K2 = value*self.mask
        
    # Temperature
    @property
    def Temp(self):
        return self.__Temp

    @Temp.setter
    def Temp(self, value):
        """
        Set a property `Temperature` [unit : K].
        The the property is only applied to the mask of the instance
        """
        self.__Temp = value*self.mask

        
    # Set Magnetization
    def setUniformMagnetization(self, thetaM0=0, phiM0=0):
        """
        Set initial magnetization to a uniform one directional state.
        :params:
            thetaM0          - Required  : polar angle of the uniform magnetization [unit: deg] (float)
            phiM0            - Required  : azimuthal angle of the uniform magnetization [unit: deg] (float)
        """
        self.thetaM0 = thetaM0
        self.phiM0 = phiM0
        thetaM0_ = self.thetaM0*degree
        phiM0_ = self.phiM0*degree

        self.mx = np.zeros(shape=(self.mask.shape))
        self.my = np.zeros(shape=(self.mask.shape))
        self.mz = np.zeros(shape=(self.mask.shape))
        
        
        self.mx[:] = self.mask*np.sin(thetaM0_)*np.cos(phiM0_)
        self.my[:] = self.mask*np.sin(thetaM0_)*np.sin(phiM0_)
        self.mz[:] = self.mask*np.cos(thetaM0_)

    def setRandomMagnetization(self):
        """
        Set initial magnetization to a random state.
        """
        self.thetaM0 = np.random.uniform(0, 180, size=self.mask.shape)
        self.phiM0 = np.random.uniform(0, 360, size=self.mask.shape)
        thetaM0_ = self.thetaM0*degree
        phiM0_ = self.phiM0*degree
        
        self.mx = np.zeros(shape=(self.mask.shape))
        self.my = np.zeros(shape=(self.mask.shape))
        self.mz = np.zeros(shape=(self.mask.shape))
        
        self.mx = self.mask*np.sin(thetaM0_)*np.cos(phiM0_)
        self.my = self.mask*np.sin(thetaM0_)*np.sin(phiM0_)
        self.mz = self.mask*np.cos(thetaM0_)


    # Merge and comcat
    def concat(self, ferromagnet, axis=0):
        """
        Concatenate another ferromagnet object along an axis.
        :params:
            ferromagnet      - Required  : another ferromagnet object.
            axis             - Required  : concatenate axis (z : 0, y: 1, x: 2).
        """
        f1 = self.clone()
        f2 = ferromagnet.clone()
        if can_concatenated(f1, f2, axis=axis)==False:
            raise ValueError("The two object can`t be concatenated.")
        
        if axis==0:
            f1.Lz = f1.Lz + f2.Lz
        if axis==1:
            f1.Ly = f1.Ly + f2.Ly
        if axis==2:
            f1.Lx = f1.Lx + f2.Lx
        f1.mask = np.concatenate((f1.mask, f2.mask), axis=axis)
        f1.generateMask()
        f1.nz, f1.ny, f1.nx = f1.mask.shape
        f1.Ms = np.concatenate((f1.Ms, f2.Ms), axis=axis)
        f1.Aex = np.concatenate((f1.Aex, f2.Aex), axis=axis)
        f1.DDMI = np.concatenate((f1.DDMI, f2.DDMI), axis=axis)
        f1.gamma = np.concatenate((f1.gamma, f2.gamma), axis=axis)
        f1.alpha = np.concatenate((f1.alpha, f2.alpha), axis=axis)
        f1.Ku = np.concatenate((f1.Ku, f2.Ku), axis=axis)
        f1.thetaK = np.concatenate((f1.thetaK, f2.thetaK), axis=axis)
        f1.phiK = np.concatenate((f1.phiK, f2.phiK), axis=axis)
        f1.mx = np.concatenate((f1.mx, f2.mx), axis=axis)
        f1.my = np.concatenate((f1.my, f2.my), axis=axis)
        f1.mz = np.concatenate((f1.mz, f2.mz), axis=axis)
        f1.Temp = np.concatenate((f1.Temp, f2.Temp), axis=axis)
        return f1

    def merge(self, ferromagnet):
        """
        Merge another ferromagnet object, which is isolated from the ferromagnet object, along an axis.
        :params:
            ferromagnet      - Required  : another ferromagnet object.
        """
        if are_space_the_same(self, ferromagnet)==False:
            raise ValueError("The space parameters of the two object are not the same.")
        if are_overlapped(self, ferromagnet)==True:
            raise ValueError("The two masks are overlapped.")
        f1 = self.clone()
        f2 = ferromagnet.clone()
        f1.mask += f2.mask
        f1.generateMask()
        f1.Ms += f2.Ms
        f1.Aex += f2.Aex
        f1.DDMI += f1.DDMI
        f1.gamma += f2.gamma
        f1.alpha += f2.alpha
        f1.Ku += f2.Ku
        f1.thetaK += f2.thetaK
        f1.phiK += f2.phiK
        f1.mx += f2.mx
        f1.my += f2.my
        f1.mz += f2.mz
        f1.Temp += f2.Temp
        return f1

#############################################################################################################################################
# demagnetizing factor
#############################################################################################################################################

class DemagFactor():
    def __init__(self, magnet, PBCx=False, PBCy=False, PBCz=False):
        """
        Initialize demag factor.
        :params:
            magnet            - Required  : magnetic object you want to calculate demag factor.
            PBCx, PBCy, PBCz  - Required  : periodic boundary condition.
        """
        self.magnet = magnet

        # Real size factor
        self.Lx = magnet.Lx
        self.Ly = magnet.Ly
        self.Lz = magnet.Lz
        
        # Number of cells
        self.nx = magnet.nx
        self.ny = magnet.ny
        self.nz = magnet.nz
        
        # Cherecteristic length
        self.lcx = self.Lx / self.nx
        self.lcy = self.Ly / self.ny
        self.lcz = self.Lz / self.nz

        self.PBCx = PBCx
        self.PBCy = PBCy
        self.PBCz = PBCz
        
    def ff(self, r1, r2, r3):
        exp = 10**-18
        res = (r2/2)*(r3**2-r1**2)*np.arcsinh(r2/(np.sqrt(r1**2+r3**2)+exp))+\
                (r3/2)*(r2**2-r1**2)*np.arcsinh(r3/(np.sqrt(r1**2+r2**2)+exp))-\
                r1*r2*r3*np.arctan(r2*r3/(r1*np.sqrt(r1**2+r2**2+r3**2)+exp))+\
                (1/6)*(2*r1**2-r2**2-r3**2)*np.sqrt(r1**2+r2**2+r3**2)
        return res

    def gg(self, r1, r2, r3):
        exp = 10**-18
        res = r1*r2*r3*np.arcsinh(r3/(np.sqrt(r1**2+r2**2)+exp))+\
                (r2/6)*(3*r3**2-r2**2)*np.arcsinh(r1/(np.sqrt(r2**2+r3**2)+exp))+\
                (r1/6)*(3*r3**2-r1**2)*np.arcsinh(r2/(np.sqrt(r1**2+r3**2)+exp))-\
                (r3**3/6)*np.arctan(r1*r2/(r3*np.sqrt(r1**2+r2**2+r3**2)+exp))-\
                (r3*r2**2/2)*np.arctan(r1*r3/(r2*np.sqrt(r1**2+r2**2+r3**2)+exp))-\
                (r3*r1**2/2)*np.arctan(r2*r3/(r1*np.sqrt(r1**2+r2**2+r3**2)+exp))-\
                r1*r2*np.sqrt(r1**2+r2**2+r3**2)/3
        return res

    def F2(self,X,Y,Z,dx,dy,dz):
        return self.ff(X,Y,Z) - self.ff(X,0,Z) - self.ff(X,Y,0) + self.ff(X,0,0)

    def F1(self,X,Y,Z,dx,dy,dz):
        return self.F2(X,Y,Z,dx,dy,dz) - self.F2(X,Y-dy,Z,dx,dy,dz) - \
               self.F2(X,Y,Z-dz,dx,dy,dz) + self.F2(X,Y-dy,Z-dz,dx,dy,dz)

    def FF(self,X,Y,Z,dx,dy,dz):
        return self.F1(X,Y+dy,Z+dz,dx,dy,dz) - self.F1(X,Y+dy,Z,dx,dy,dz) - \
               self.F1(X,Y,Z+dz,dx,dy,dz) + self.F1(X,Y,Z,dx,dy,dz)

    def G2(self,X,Y,Z,dx,dy,dz):
        return self.gg(X,Y,Z) - self.gg(X,Y,0)

    def G1(self,X,Y,Z,dx,dy,dz):
        return self.G2(X+dx,Y,Z+dz,dx,dy,dz) - self.G2(X+dx,Y,Z,dx,dy,dz) - \
               self.G2(X,Y,Z+dz,dx,dy,dz) + self.G2(X,Y,Z,dx,dy,dz)

    def GG(self,X,Y,Z,dx,dy,dz):
        return self.G1(X,Y,Z,dx,dy,dz) - self.G1(X,Y-dy,Z,dx,dy,dz) - \
               self.G1(X,Y,Z-dz,dx,dy,dz) + self.G1(X,Y-dy,Z-dz,dx,dy,dz)

    def Nxx(self,X,Y,Z,dx,dy,dz):
        return (1/(4*math.pi*dx*dy*dz)) * (2*self.FF(X,Y,Z,dx,dy,dz)-self.FF(X+dx,Y,Z,dx,dy,dz)-self.FF(X-dx,Y,Z,dx,dy,dz))
    
    def Nyy(self,X,Y,Z,dx,dy,dz):
        return self.Nxx(Y,X,Z,dy,dx,dz)

    def Nzz(self,X,Y,Z,dx,dy,dz):
        return self.Nxx(Z,X,Y,dz,dx,dy)

    def Nxy(self,X,Y,Z,dx,dy,dz):
        return (1/(4*math.pi*dx*dy*dz)) * (self.GG(X,Y,Z,dx,dy,dz)-self.GG(X-dx,Y,Z,dx,dy,dz)-self.GG(X,Y+dy,Z,dx,dy,dz)+self.GG(X-dx,Y+dy,Z,dx,dy,dz))

    def Nxz(self,X,Y,Z,dx,dy,dz):
        return self.Nxy(X,Z,Y,dx,dz,dy)

    def Nyz(self,X,Y,Z,dx,dy,dz):
        return self.Nxy(Y,Z,X,dy,dz,dx)                       
    
    def calDemagFactor(self):
        """
        Calculate demag factor.
        """
        prefactor = -4*math.pi
                    
        dx = self.lcx
        dy = self.lcy
        dz = self.lcz
        a1 = (np.array(range(2*self.nx-1)) - self.nx+1)*dx
        a2 = (np.array(range(2*self.ny-1)) - self.ny+1)*dy
        a3 = (np.array(range(2*self.nz-1)) - self.nz+1)*dz

        a3, a2, a1 = np.meshgrid(a3,a2,a1, sparse=False, indexing='ij')

        print('Start calculating demag factors')
        Nxx_val = self.Nxx(a1,a2,a3,dx,dy,dz)
        Nxy_val = self.Nxy(a1,a2,a3,dx,dy,dz)
        Nxz_val = self.Nxz(a1,a2,a3,dx,dy,dz)
        Nyy_val = self.Nyy(a1,a2,a3,dx,dy,dz)
        Nyz_val = self.Nyz(a1,a2,a3,dx,dy,dz)
        Nzz_val = self.Nzz(a1,a2,a3,dx,dy,dz)
        print('End calculating demag factors')

        self.Ntotxx = prefactor*Nxx_val
        self.Ntotxy = prefactor*Nxy_val
        self.Ntotxz = prefactor*Nxz_val
        self.Ntotyy = prefactor*Nyy_val
        self.Ntotyz = prefactor*Nyz_val
        self.Ntotzz = prefactor*Nzz_val

        self.FNtotxx = np.fft.fftn(self.Ntotxx, axes=(0,1,2), norm=None)
        self.FNtotxy = np.fft.fftn(self.Ntotxy, axes=(0,1,2), norm=None)
        self.FNtotxz = np.fft.fftn(self.Ntotxz, axes=(0,1,2), norm=None)
        self.FNtotyy = np.fft.fftn(self.Ntotyy, axes=(0,1,2), norm=None)
        self.FNtotyz = np.fft.fftn(self.Ntotyz, axes=(0,1,2), norm=None)
        self.FNtotzz = np.fft.fftn(self.Ntotzz, axes=(0,1,2), norm=None)

    def calDemagField(self):
        """
        Calculate demag field.
        """
        Mx_ = self.magnet.mx * self.magnet.Ms
        My_ = self.magnet.my * self.magnet.Ms
        Mz_ = self.magnet.mz * self.magnet.Ms

        self.padMx = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))
        self.padMy = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))
        self.padMz = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))

        self.padMx[0:self.nz, 0:self.ny, 0:self.nx] = Mx_
        self.padMy[0:self.nz, 0:self.ny, 0:self.nx] = My_
        self.padMz[0:self.nz, 0:self.ny, 0:self.nx] = Mz_

        if self.PBCx:
            self.padMx[0:self.nz, 0:self.ny, self.nx:] = Mx_[:,:,:-1]
            self.padMy[0:self.nz, 0:self.ny, self.nx:] = My_[:,:,:-1]
            self.padMz[0:self.nz, 0:self.ny, self.nx:] = Mz_[:,:,:-1]        

        if self.PBCy:
            self.padMx[0:self.nz, self.ny:, 0:self.nx] = Mx_[:,:-1,:]
            self.padMy[0:self.nz, self.ny:, 0:self.nx] = My_[:,:-1,:]
            self.padMz[0:self.nz, self.ny:, 0:self.nx] = Mz_[:,:-1,:]

        if self.PBCz:
            self.padMx[self.nz:, 0:self.ny, 0:self.nx] = Mx_[:-1,:,:]
            self.padMy[self.nz:, 0:self.ny, 0:self.nx] = My_[:-1,:,:]
            self.padMz[self.nz:, 0:self.ny, 0:self.nx] = Mz_[:-1,:,:]

        if self.PBCx & self.PBCy:
            self.padMx[0:self.nz, self.ny:, self.nx:] = Mx_[:,:-1,:-1]
            self.padMy[0:self.nz, self.ny:, self.nx:] = My_[:,:-1,:-1]
            self.padMz[0:self.nz, self.ny:, self.nx:] = Mz_[:,:-1,:-1]

        if self.PBCy & self.PBCz:
            self.padMx[self.nz:, self.ny:, 0:self.nx] = Mx_[:-1,:-1,:]
            self.padMy[self.nz:, self.ny:, 0:self.nx] = My_[:-1,:-1,:]
            self.padMz[self.nz:, self.ny:, 0:self.nx] = Mz_[:-1,:-1,:]

        if self.PBCz & self.PBCx:
            self.padMx[self.nz:, 0:self.ny, self.nx:] = Mx_[:-1,:,:-1]
            self.padMy[self.nz:, 0:self.ny, self.nx:] = My_[:-1,:,:-1]
            self.padMz[self.nz:, 0:self.ny, self.nx:] = Mz_[:-1,:,:-1]

        if self.PBCx & self.PBCy & self.PBCz:
            self.padMx[self.nz:, self.ny:, self.nx:] = Mx_[:-1,:-1,:-1]
            self.padMy[self.nz:, self.ny:, self.nx:] = My_[:-1,:-1,:-1]
            self.padMz[self.nz:, self.ny:, self.nx:] = Mz_[:-1,:-1,:-1]

        self.FpadMx = np.fft.fftn(self.padMx, axes=(0,1,2), norm=None)
        self.FpadMy = np.fft.fftn(self.padMy, axes=(0,1,2), norm=None)
        self.FpadMz = np.fft.fftn(self.padMz, axes=(0,1,2), norm=None)

        self.Hdemagx = np.real(np.fft.ifftn(self.FNtotxx*self.FpadMx + self.FNtotxy*self.FpadMy + self.FNtotxz*self.FpadMz, axes=(0,1,2), norm=None))
        self.Hdemagy = np.real(np.fft.ifftn(self.FNtotxy*self.FpadMx + self.FNtotyy*self.FpadMy + self.FNtotyz*self.FpadMz, axes=(0,1,2), norm=None))
        self.Hdemagz = np.real(np.fft.ifftn(self.FNtotxz*self.FpadMx + self.FNtotyz*self.FpadMy + self.FNtotzz*self.FpadMz, axes=(0,1,2), norm=None))

        self.hdemagx = self.Hdemagx[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]
        self.hdemagy = self.Hdemagy[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]
        self.hdemagz = self.Hdemagz[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]

    def chech_Ntot(self):
        """
        Check demag factor.
        """
        Ms_ = np.abs(self.magnet.Ms).max()
        for direction, thetaM0, phiM0 in [('x', 90, 0), ('y', 90, 90), ('z', 0, 0)]:

            self.magnet.setUniformMagnetization(thetaM0=thetaM0, phiM0=phiM0)
            self.calDemagField()

#             Nx_ = sum((self.hdemagx*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)
#             Ny_ = sum((self.hdemagy*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)
#             Nz_ = sum((self.hdemagz*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)
            
            prefactorMs = 4*math.pi*Ms_
            mask_area = (self.magnet.mask).sum()
            Nx_ = (self.hdemagx*self.magnet.mask).sum()/mask_area / prefactorMs
            Ny_ = (self.hdemagy*self.magnet.mask).sum()/mask_area / prefactorMs
            Nz_ = (self.hdemagz*self.magnet.mask).sum()/mask_area / prefactorMs

            
            
            print('Ms direction = {}'.format(direction), ' :::: [ Hdemagx/4piMs = {:.4f}, H_demagy/4piMs = {:.4f}, H_demagz/4piMs = {:.4f} ]'.format(
            Nx_.tolist(),
            Ny_.tolist(),
            Nz_.tolist()))

#############################################################################################################################################
# demagnetizing factor
#############################################################################################################################################

class Evolver():
    """
    Evolver updates magnetization state of the magnetic object.
    """
    def __init__(self, magnet=None, tstep=10**-15, 
                 uniaxial_anisotropy_field=True,
                 exchang_field=True,
                 DMI_field=False,
                 demag_field=True,
                 thermal_field=True,
                 external_field=False,
                 FLT_field=False,
                 DLT_field=False,
                 warm_up=True,
                 random_state=42):
        """
        Initialize evolver
        :params:
            magnet                    - Required  : magnetic object of interest (object)
            tstep                     - Required  : time step (float)
            uniaxial_anisotropy_field - Required  : whether the system contains Ku or not (bool)
            exchang_field             - Required  : whether the system contains Aex or not (bool)
            DMI_field                 - Required  : whether the system contains DMI or not (bool)
            demag_field               - Required  : whether the system contains demag or not (bool)
            thermal_field             - Required  : whether the system contains thermal or not (bool)
            external_field            - Required  : whether the system contains external or not (bool)
            FLT_field                 - Required  : whether the system contains FLT or not (bool)
            DLT_field                 - Required  : whether the system contains DLT or not (bool)
            warm_up                   - Required  : whether the stage is for warm_up or not. if warm_up=True, precession is ignored (bool)
            random_state              - Required  : random_state for thermal agitation field (int)
        """
        self.magnet = magnet
        self.tstep = tstep
        self.time = 0

        self.uniaxial_anisotropy_field = uniaxial_anisotropy_field
        self.exchang_field = exchang_field
        self.external_field = external_field
        self.DMI_field = DMI_field
        self.demag_field = demag_field
        self.thermal_field = thermal_field
        self.random_state = random_state
        self.warm_up = warm_up

        gamma_ = self.magnet.gamma
        alpha_ = self.magnet.alpha
        self.C1 = gamma_/(1+alpha_**2) 
        self.C2 = alpha_*self.C1

        self.demagFactor = DemagFactor(self.magnet,
                                       PBCx = self.magnet.PBCx,
                                       PBCy = self.magnet.PBCy,
                                       PBCz = self.magnet.PBCz)
        
        self.demagFactor.calDemagFactor()
        
        # Initialize External magnetic field
        self.hextx = 0
        self.hexty = 0
        self.hextz = 0

    def setUniformExternalMagneticField(self, Hextx, Hexty, Hextz):
        """
        Set uniform external magnetic field.
        :params:
            Hextx, Hexty, Hextz  - Required  : external magnetic field.
        """
        self.hextx = Hextx
        self.hexty = Hexty
        self.hextz = Hextz

    def Huniaxialanisotropy(self):
        """
        Calculate uniaxial anisotropy field.
        """
        mx_ = self.magnet.mx
        my_ = self.magnet.my
        mz_ = self.magnet.mz

        Ku_ = self.magnet.Ku
        thetaK_ = self.magnet.thetaK*degree
        phiK_ = self.magnet.phiK*degree
        
        Kux_ = Ku_*np.sin(thetaK_)*np.cos(phiK_)
        Kuy_ = Ku_*np.sin(thetaK_)*np.sin(phiK_)
        Kuz_ = Ku_*np.cos(thetaK_)

        Hux_ = 2 * Kux_ / (self.magnet.Ms + 10**-10) * self.magnet.mask
        Huy_ = 2 * Kuy_ / (self.magnet.Ms + 10**-10) * self.magnet.mask
        Huz_ = 2 * Kuz_ / (self.magnet.Ms + 10**-10) * self.magnet.mask

        Hu_ = Hux_ * mx_ + Huy_ * my_ + Huz_ * mz_

        
        self.hux = Hu_ * np.sin(thetaK_)*np.cos(phiK_)
        self.huy = Hu_ * np.sin(thetaK_)*np.sin(phiK_)
        self.huz = Hu_ * np.cos(thetaK_)

    def Hexchange(self):
        """
        Calculate exchange field.
        """
        Aex_ = self.magnet.Aex
        Ms_ = self.magnet.Ms
        Hex_ = 2 * Aex_ / (Ms_ + 10**-100) * self.magnet.mask

        mx_ = self.magnet.mx
        my_ = self.magnet.my
        mz_ = self.magnet.mz
        
        dx_ = self.magnet.lcx
        dy_ = self.magnet.lcy
        dz_ = self.magnet.lcz

        del2mms = []
        for i, mm in enumerate([mx_, my_, mz_]):
            mm_zp = np.roll(mm, shift=+1, axis=0)
            mm_zm = np.roll(mm, shift=-1, axis=0)
            mm_yp = np.roll(mm, shift=+1, axis=1)
            mm_ym = np.roll(mm, shift=-1, axis=1)
            mm_xp = np.roll(mm, shift=+1, axis=2)
            mm_xm = np.roll(mm, shift=-1, axis=2)

            d2_mm_per_dx2 = (mm_xp - mm) * ~(self.magnet.mask_xp) - (mm - mm_xm) * ~(self.magnet.mask_xm)
            d2_mm_per_dy2 = (mm_yp - mm) * ~(self.magnet.mask_yp) - (mm - mm_ym) * ~(self.magnet.mask_ym)
            d2_mm_per_dz2 = (mm_zp - mm) * ~(self.magnet.mask_zp) - (mm - mm_zm) * ~(self.magnet.mask_zm)

            d2_mm_per_dx2 = d2_mm_per_dx2 / dx_**2
            d2_mm_per_dy2 = d2_mm_per_dy2 / dy_**2
            d2_mm_per_dz2 = d2_mm_per_dz2 / dz_**2
            
            del2mm = (d2_mm_per_dx2 + d2_mm_per_dy2 + d2_mm_per_dz2) * self.magnet.mask
            del2mms.append(del2mm)
      
        hexx, hexy, hexz = Hex_ * del2mms[0], Hex_ * del2mms[1], Hex_ * del2mms[2]

        self.hexx = hexx
        self.hexy = hexy
        self.hexz = hexz

    def Hdemag(self):
        """
        Calculate demag field.
        """
        self.demagFactor.calDemagField()
        self.hdemagx = self.demagFactor.hdemagx
        self.hdemagy = self.demagFactor.hdemagy
        self.hdemagz = self.demagFactor.hdemagz

    def Hthermal(self, random_state=42):
        """
        Calculate thermal agitation.
        """
        T_ = self.magnet.Temp
        alpha_ = self.magnet.alpha
        gamma_ = self.magnet.gamma
        Ms_ = self.magnet.Ms
        dx_ = self.magnet.lcx
        dy_ = self.magnet.lcy
        dz_ = self.magnet.lcz

        H_var = 2*alpha_*kB*T_ / (gamma_*Ms_*dx_*dy_*dz_*self.tstep + 10**-18) * self.magnet.mask
        H_sigma = np.sqrt(H_var)

        np.random.seed(random_state)        
        self.hthx = np.random.randn(H_sigma.size).reshape(H_sigma.shape) * H_sigma
        self.hthy = np.random.randn(H_sigma.size).reshape(H_sigma.shape) * H_sigma
        self.hthz = np.random.randn(H_sigma.size).reshape(H_sigma.shape) * H_sigma
        
    def Hexchange_with_DMI(self):
        """
        Calculate exchange + DMI field. This method is currently under development.
        """
        DDMI_ = self.magnet.DDMI
        Aex_ = self.magnet.Aex
        Ms_ = self.magnet.Ms

        DperA_ = 0.5 * DDMI_ / (Aex_  + 10**-100) * self.magnet.mask
        HDMI_ = DDMI_ / (Ms_ + 10**-100) * self.magnet.mask

        mx_ = self.magnet.mx
        my_ = self.magnet.my
        mz_ = self.magnet.mz

    def HDLT(self):
        """
        Calculate damping-like spin-orbit torque field. This method is currently under development.
        """
        pass

    def HFLT(self):
        """
        Calculate field-like spin-orbit torque  field. This method is currently under development.
        """
        pass

    def LLG(self, mx, my, mz, heffx, heffy, heffz):
        """
        LLG equation.
        """
        m_ = np.array([[mx],[my],[mz]])
        heff_ = np.array([[heffx], [heffy], [heffz]])
        rk_ = - self.C1 * MatrixCrossProduct(m_, heff_) - self.C2 * MatrixCrossProduct(m_, MatrixCrossProduct(m_, heff_))
        rkx_ = rk_[0,0]
        rky_ = rk_[1,0]
        rkz_ = rk_[2,0]
        return rkx_, rky_, rkz_

    def LLG_warmup(self, mx, my, mz, heffx, heffy, heffz):
        """
        LLG equation without the precession term.
        This equation is used for warm_up stage.
        """
        m_ = np.array([[mx],[my],[mz]])
        heff_ = np.array([[heffx], [heffy], [heffz]])
        rk_ = - self.C2 * MatrixCrossProduct(m_, MatrixCrossProduct(m_, heff_))
        rkx_ = rk_[0,0]
        rky_ = rk_[1,0]
        rkz_ = rk_[2,0]
        return rkx_, rky_, rkz_

    def Heff(self):
        """
        Sum all magnetic fields.
        """

        # Initialize heff
        self.heffx_ = np.zeros(shape=self.magnet.mask.shape)
        self.heffy_ = np.zeros(shape=self.magnet.mask.shape)
        self.heffz_ = np.zeros(shape=self.magnet.mask.shape)

        # Add hext
        if self.external_field:
            self.heffx_ += self.hextx
            self.heffy_ += self.hexty
            self.heffz_ += self.hextz
            
            
        # Add hu
        if self.uniaxial_anisotropy_field:
            self.Huniaxialanisotropy()
            self.heffx_ += self.hux
            self.heffy_ += self.huy
            self.heffz_ += self.huz
        
        # Add hex
        if self.exchang_field:
            if ~self.DMI_field:
                self.Hexchange()
            if self.DMI_field:
                self.Hexchange_with_DMI()

            self.heffx_ += self.hexx
            self.heffy_ += self.hexy
            self.heffz_ += self.hexz
      

        # Add hdemag
        if self.demag_field:
            self.Hdemag()
            self.heffx_ += self.hdemagx
            self.heffy_ += self.hdemagy
            self.heffz_ += self.hdemagz

        # Add hth
        if self.thermal_field:
            self.Hthermal(random_state=self.random_state)
            self.heffx_ += self.hthx
            self.heffy_ += self.hthy
            self.heffz_ += self.hthz


    def cal4thRK(self, equation=None):
        """
        calculate llg equation by using 4th-order Runge-Kutta method.
        """
        k1x_, k1y_, k1z_ = equation(self.magnet.mx,
                                    self.magnet.my,
                                    self.magnet.mz, 
                                    self.heffx_, 
                                    self.heffy_, 
                                    self.heffz_)
        
        k2x_, k2y_, k2z_ = equation(self.magnet.mx + 0.5*self.tstep * k1x_,
                                    self.magnet.my + 0.5*self.tstep * k1y_,
                                    self.magnet.mz + 0.5*self.tstep * k1z_,
                                    self.heffx_, 
                                    self.heffy_, 
                                    self.heffz_)
        
        k3x_, k3y_, k3z_ = equation(self.magnet.mx + 0.5*self.tstep * k2x_,
                                    self.magnet.my + 0.5*self.tstep * k2y_,
                                    self.magnet.mz + 0.5*self.tstep * k2z_,
                                    self.heffx_, 
                                    self.heffy_, 
                                    self.heffz_)
        
        k4x_, k4y_, k4z_ = equation(self.magnet.mx + self.tstep * k3x_,
                                    self.magnet.my + self.tstep * k3y_,
                                    self.magnet.mz + self.tstep * k3z_,
                                    self.heffx_, 
                                    self.heffy_, 
                                    self.heffz_)
        
        self.dmx = self.tstep * (k1x_ + 2*k2x_ + 2*k3x_ + k4x_)/6
        self.dmy = self.tstep * (k1y_ + 2*k2y_ + 2*k3y_ + k4y_)/6
        self.dmz = self.tstep * (k1z_ + 2*k2z_ + 2*k3z_ + k4z_)/6

    def evolve(self):
        """
        Update self.magnet.
        """
        self.time += self.tstep
        
        self.Heff()

        if self.warm_up:
            self.cal4thRK(equation=self.LLG_warmup)
        else:
            self.cal4thRK(equation=self.LLG)

        self.magnet.mx += self.dmx
        self.magnet.my += self.dmy
        self.magnet.mz += self.dmz

        norm = np.sqrt(self.magnet.mx**2+self.magnet.my**2+self.magnet.mz**2) +10**-100

        self.magnet.mx /= norm
        self.magnet.my /= norm
        self.magnet.mz /= norm

        self.demagFactor.magnet = self.magnet
        
        self.dm = np.sqrt(self.dmx**2 + self.dmy**2 + self.dmz**2).flatten().max().tolist()

    def cal_tstep(self, ideal_dm=0.01):
        """
        Calculate optimal time step for the next evolve.
        """
        self.tstep *= (ideal_dm/self.dm)
        self.tstep = self.tstep

#     def cal_stop(self, ideal_dm=1.0*10**-15):
#         """
#         Stop calculation were dm < ideal_dm.
#         """
#         dm = np.sqrt(self.dmx**2 + self.dmy**2 + self.dmz**2).flatten().max()
#         if dm < ideal_dm:
#             print('Equilibrium state is achieved now')
#             break

#############################################################################################################################################
# scheduler
#############################################################################################################################################

class scheduler():
    def Hfunction(self, Hx_f = 0, Hy_f = 0, Hz_f = 0):
        self.Hx_f = Hx_f
        self.Hy_f = Hy_f
        self.Hz_f = Hz_f

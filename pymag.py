import cupy as np
import math
import copy
import itertools


"""
Default valeus
hbar = 1.05457266*10**-27  # [unit : erg s]
echarge = 1.60217662*10-19 # [unit : A s]
kB = 1.380648*10**-16  # [unit : erg / K]
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


def MatrixCrossProduct(a, b):
  c = []
  c.append(a[1]*b[2]-a[2]*b[1])
  c.append(a[2]*b[0]-a[0]*b[2])
  c.append(a[0]*b[1]-a[1]*b[0])
  c = np.array(c)
  return c
  
class MagneticObject():
    def __init__(self, 
                 Lx=100*10**-7, Ly=100*10**-7, Lz=5*10**-7, 
                 nx=100, ny=100, nz=5, 
                 PBCx=False, PBCy=False, PBCz=False):
        
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
        self.mask_zp = np.roll(self.mask, shift=+1, axis=0)
        self.mask_zm = np.roll(self.mask, shift=-1, axis=0)
        self.mask_yp = np.roll(self.mask, shift=+1, axis=1)
        self.mask_ym = np.roll(self.mask, shift=-1, axis=1)
        self.mask_xp = np.roll(self.mask, shift=+1, axis=2)
        self.mask_xm = np.roll(self.mask, shift=-1, axis=2)
        
        self.mask_zp = (self.mask - self.mask_zp)*self.mask
        self.mask_zm = (self.mask - self.mask_zm)*self.mask
        self.mask_yp = (self.mask - self.mask_yp)*self.mask
        self.mask_ym = (self.mask - self.mask_ym)*self.mask
        self.mask_xp = (self.mask - self.mask_xp)*self.mask
        self.mask_xm = (self.mask - self.mask_xm)*self.mask
        
        self.mask_zp = self.mask_zp.astype(bool)
        self.mask_zm = self.mask_zm.astype(bool)
        self.mask_yp = self.mask_yp.astype(bool)
        self.mask_ym = self.mask_ym.astype(bool)
        self.mask_xp = self.mask_xp.astype(bool)
        self.mask_xm = self.mask_xm.astype(bool)
        
        self.mask_edge = self.mask_zp | self.mask_zm | self.mask_yp | self.mask_ym | self.mask_xp | self.mask_xm
        self.mask_face = self.mask.astype(bool) & ~self.mask_edge
        
    def clone(self):
        return copy.deepcopy(self)
    
    def setCylindricalMask(self, 
                           center_x=50*10**-7, center_y=50*10**-7, center_z=2.5*10**-7, 
                           radius_x=40*10**-7, radius_y=40*10**-7, 
                           angle=0, height=4*10**-7):
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

class Ferromagnet(MagneticObject):
    def __init__(self, **kwargs):
        super(Ferromagnet, self).__init__(**kwargs)

    # Saturation magnetization
    @property
    def Ms(self):
        return self.__Ms

    @Ms.setter
    def Ms(self, value):
        self.__Ms = value*self.mask

    # Exchagne stiffness
    @property
    def Aex(self):
        return self.__Aex

    @Aex.setter
    def Aex(self, value):
        self.__Aex = value*self.mask

    # Gyromagnetic ratio
    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, value):
        self.__gamma = value*self.mask

    # Damping constant
    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        self.__alpha = value*self.mask

    # DMI
    @property
    def DDMI(self):
        return self.__DDMI

    @DDMI.setter
    def DDMI(self, value):
        self.__DDMI = value*self.mask

    # Uniaxial anisotropy
    @property
    def Ku(self):
        return self.__Ku

    @Ku.setter
    def Ku(self, value):
        self.__Ku = value*self.mask

    @property
    def thetaK(self):
        return self.__thetaK

    @thetaK.setter
    def thetaK(self, value):
        self.__thetaK = value*self.mask

    @property
    def phiK(self):
        return self.__phiK

    @phiK.setter
    def phiK(self, value):
        self.__phiK = value*self.mask

    # Temperature
    @property
    def Temp(self):
        return self.__Temp

    @Temp.setter
    def Temp(self, value):
        self.__Temp = value*self.mask

        
    # Set Magnetization
    def setUniformMagnetization(self, thetaM0=0, phiM0=0):
        self.thetaM0 = thetaM0
        self.phiM0 = phiM0

        self.mx = np.zeros(shape=(self.mask.shape))
        self.my = np.zeros(shape=(self.mask.shape))
        self.mz = np.zeros(shape=(self.mask.shape))
        
        degree = math.pi/180
        thetaM0_ = self.thetaM0*degree
        phiM0_ = self.phiM0*degree
        self.mx[:] = self.mask*np.sin(thetaM0_)*np.cos(phiM0_)
        self.my[:] = self.mask*np.sin(thetaM0_)*np.sin(phiM0_)
        self.mz[:] = self.mask*np.cos(thetaM0_)

    def setRandomMagnetization(self):
        self.thetaM0 = np.random.uniform(0, 180, size=self.mask.shape)
        self.phiM0 = np.random.uniform(0, 360, size=self.mask.shape)
        
        self.mx = np.zeros(shape=(self.mask.shape))
        self.my = np.zeros(shape=(self.mask.shape))
        self.mz = np.zeros(shape=(self.mask.shape))
        
        degree = math.pi/180
        thetaM0_ = self.thetaM0*degree
        phiM0_ = self.phiM0*degree
        self.mx = self.mask*np.sin(thetaM0_)*np.cos(phiM0_)
        self.my = self.mask*np.sin(thetaM0_)*np.sin(phiM0_)
        self.mz = self.mask*np.cos(thetaM0_)

    def concat(self, ferromagnet, axis=0):
        f1 = self.clone()
        f2 = ferromagnet.clone()
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

class DemagFactor():
    def __init__(self, magnet, PBCx=False, PBCy=False, PBCz=False):
        
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
      Ms_ = np.abs(self.magnet.Ms).max()
      for direction, thetaM0, phiM0 in [('x', 90, 0), ('y', 90, 90), ('z', 0, 0)]:

        self.magnet.setUniformMagnetization(thetaM0=thetaM0, phiM0=phiM0)
        self.calDemagField()


        Nx_ = sum((self.hdemagx*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)
        Ny_ = sum((self.hdemagy*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)
        Nz_ = sum((self.hdemagz*self.magnet.mask).flatten())/sum((self.magnet.mask).flatten()) / (4*math.pi*Ms_)



        print('Ms direction = {}'.format(direction), ' :::: [ Hdemagx/4piMs = {:.4f}, H_demagy/4piMs = {:.4f}, H_demagz/4piMs = {:.4f} ]'.format(
          Nx_.tolist(),
          Ny_.tolist(),
          Nz_.tolist()))
        
class Evolver():
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
        self.magnet = magnet
        self.tstep = tstep
        self.time = 0

        self.uniaxial_anisotropy_field = uniaxial_anisotropy_field
        self.exchang_field = exchang_field
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

    def Huniaxialanisotropy(self):
        mx_ = self.magnet.mx
        my_ = self.magnet.my
        mz_ = self.magnet.mz

        Ku_ = self.magnet.Ku
        thetaK_ = self.magnet.thetaK
        phiK_ = self.magnet.phiK
        
        Kux_ = Ku_*np.sin(thetaK_)*np.cos(phiK_)
        Kuy_ = Ku_*np.sin(thetaK_)*np.sin(phiK_)
        Kuz_ = Ku_*np.cos(thetaK_)

        Hux_ = 2 * Kux_ / (self.magnet.Ms + 10**-10) * self.magnet.mask
        Huy_ = 2 * Kuy_ / (self.magnet.Ms + 10**-10) * self.magnet.mask
        Huz_ = 2 * Kuz_ / (self.magnet.Ms + 10**-10) * self.magnet.mask

        Hu_ = Hux_ * mx_ + Huy_ * my_ + Huz_ * mz_

        degree = math.pi/180
        thetaK_ = thetaK_*degree
        phiK_ = phiK_*degree
        self.hux = Hu_ * np.sin(thetaK_)*np.cos(phiK_)
        self.huy = Hu_ * np.sin(thetaK_)*np.sin(phiK_)
        self.huz = Hu_ * np.cos(thetaK_)

    def Hexchange(self):
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
      self.demagFactor.calDemagField()
      self.hdemagx = self.demagFactor.hdemagx
      self.hdemagy = self.demagFactor.hdemagy
      self.hdemagz = self.demagFactor.hdemagz

    def Hthermal(self, random_state=42):
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
        DDMI_ = self.magnet.DDMI
        Aex_ = self.magnet.Aex
        Ms_ = self.magnet.Ms

        DperA_ = 0.5 * DDMI_ / (Aex_  + 10**-100) * self.magnet.mask
        HDMI_ = DDMI_ / (Ms_ + 10**-100) * self.magnet.mask

        mx_ = self.magnet.mx
        my_ = self.magnet.my
        mz_ = self.magnet.mz

    def HDLT(self):
      pass

    def HFLT(self):
      pass

    def LLG(self, mx, my, mz, heffx, heffy, heffz):
        m_ = np.array([[mx],[my],[mz]])
        heff_ = np.array([[heffx], [heffy], [heffz]])
        rk_ = - self.C1 * MatrixCrossProduct(m_, heff_) - self.C2 * MatrixCrossProduct(m_, MatrixCrossProduct(m_, heff_))
        rkx_ = rk_[0,0]
        rky_ = rk_[1,0]
        rkz_ = rk_[2,0]
        return rkx_, rky_, rkz_

    def LLG_warmup(self, mx, my, mz, heffx, heffy, heffz):
        m_ = np.array([[mx],[my],[mz]])
        heff_ = np.array([[heffx], [heffy], [heffz]])
        rk_ = - self.C2 * MatrixCrossProduct(m_, MatrixCrossProduct(m_, heff_))
        rkx_ = rk_[0,0]
        rky_ = rk_[1,0]
        rkz_ = rk_[2,0]
        return rkx_, rky_, rkz_

    def Heff(self):
        # Initialize heff
        self.heffx_ = np.zeros(shape=self.magnet.mask.shape)
        self.heffy_ = np.zeros(shape=self.magnet.mask.shape)
        self.heffz_ = np.zeros(shape=self.magnet.mask.shape)

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

    def cal_tstep(self, ideal_dm=0.01):
        dm = np.sqrt(self.dmx**2 + self.dmy**2 + self.dmz**2).flatten().max()
        self.tstep *= (ideal_dm/dm)

class scheduler():
  def Hfunction(self, Hx_f = 0, Hy_f = 0, Hz_f = 0):
    self.Hx_f = Hx_f
    self.Hy_f = Hy_f
    self.Hz_f = Hz_f

import cupy as np
from . import MagneticObject

class Electrode(MagneticObject):
    def __init__(self, magnet, PBCx=False, PBCy=False, PBCz=False, **kwargs):
        super(Electrode).__init__(**kwargs)

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

    def setCuboidMask(self,
                      center_x=50*10**-7, center_y=50*10**-7, center_z=2.5*10**-7, 
                      l_x=40*10**-7, l_y=40*10**-7, l_z=4*10**-7):
        
        self.center_x = center_x # x axis position of cylindrical geometry [unit: m]
        self.center_y = center_y # y axis position of cylindrical geometry [unit: m]
        self.center_z = center_z # z axis position of cylindrical geometry [unit: m]
        self.l_x = l_x 
        self.l_y = l_y 
        self.l_z = l_z 
        
        Z, Y, X = np.ogrid[:self.nz, :self.ny, :self.nx]
        X = (X + 0.5) * self.lcx - self.center_x
        Y = (Y + 0.5) * self.lcy - self.center_y
        Z = (Z + 0.5) * self.lcz - self.center_z
        
        X = X.round(decimals=16)
        Y = Y.round(decimals=16)
        Z = Z.round(decimals=16)

        self.mask = ( np.abs(X) < l_x/2 ) & ( np.abs(Y) < l_y/2 ) & ( np.abs(Z) < l_z/2 )
        self.generateMask()

    def setElbowMask(self, center_x=50*10**-7, center_y=50*10**-7, center_z=2.5*10**-7, 
                     r_s=15*10**-7, r_l=20*10**-7, l_z=4*10**-7,
                     phi_start=0, phi_end=90):
        self.center_x = center_x # x axis position of cylindrical geometry [unit: m]
        self.center_y = center_y # y axis position of cylindrical geometry [unit: m]
        self.center_z = center_z # z axis position of cylindrical geometry [unit: m]
        self.r_s = r_s 
        self.r_l = r_l 
        self.l_z = l_z
        self.phi_start = phi_start
        self.phi_end = phi_end
        
        Z, Y, X = np.ogrid[:self.nz, :self.ny, :self.nx]
        X = (X + 0.5) * self.lcx - self.center_x
        Y = (Y + 0.5) * self.lcy - self.center_y
        Z = (Z + 0.5) * self.lcz - self.center_z
        
        X = X.round(decimals=16)
        Y = Y.round(decimals=16)
        Z = Z.round(decimals=16)

        self.phiJ = np.arctan2(Y,X)

        deg = math.pi/180

        self.mask = ( X**2 + Y**2 > r_s**2 ) & ( X**2 + Y**2 < r_l**2 ) & ( np.abs(Z) < l_z/2 ) & (np.arctan2(Y,X) > phi_start*deg) & (np.arctan2(Y,X) < phi_end*deg)
        self.generateMask()

    def setUniformCurrentDensity(self, Jx=0, Jy=0, Jz=0):
        self.Jx = Jx * self.mask
        self.Jy = Jy * self.mask
        self.Jz = Jz * self.mask

    def setElbowCurrentDensity(self, J):
        self.Jx = -np.sin(self.phiJ) * J * self.mask
        self.Jy =  np.cos(self.phiJ) * J * self.mask
        self.Jz = 0 * self.mask

    def Kxy(self,x,y,z,dx,dy,dz):
        R = np.sqrt(x**2 + y**2 + z**2)
        return   dx * dy * dz * z / (R**3 + 10**-18)

    def Kxz(self,x,y,z,dx,dy,dz):
        R = np.sqrt(x**2 + y**2 + z**2)
        return - dx * dy * dz * y / (R**3 + 10**-18)

    def Kyz(self,x,y,z,dx,dy,dz):
        R = np.sqrt(x**2 + y**2 + z**2)
        return   dx * dy * dz * x / (R**3 + 10**-18)

    def calHampkernel(self):
        prefactor = 1/(4*math.pi)
        dx = self.lcx
        dy = self.lcy
        dz = self.lcz

        a1 = (np.array(range(2*self.nx-1)) - self.nx+1)*dx
        a2 = (np.array(range(2*self.ny-1)) - self.ny+1)*dy
        a3 = (np.array(range(2*self.nz-1)) - self.nz+1)*dz
 
        a3, a2, a1 = np.meshgrid(a3,a2,a1, sparse=False, indexing='ij')
 
        print('Start calculating demag factors')
        Kxy_val = self.Kxy(a1,a2,a3,dx,dy,dz)
        Kxz_val = self.Kxz(a1,a2,a3,dx,dy,dz)
        Kyz_val = self.Kyz(a1,a2,a3,dx,dy,dz)
        print('End calculating demag factors')
 
        self.Ktotxy = prefactor*Kxy_val
        self.Ktotxz = prefactor*Kxz_val
        self.Ktotyz = prefactor*Kyz_val
        
        self.FKtotxy = np.fft.fftn(self.Ktotxy, axes=(0,1,2), norm=None)
        self.FKtotxz = np.fft.fftn(self.Ktotxz, axes=(0,1,2), norm=None)
        self.FKtotyz = np.fft.fftn(self.Ktotyz, axes=(0,1,2), norm=None)

        
    def calHamp(self):

        Jx_ = self.Jx
        Jy_ = self.Jy
        Jz_ = self.Jz

        self.padJx = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))
        self.padJy = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))
        self.padJz = np.zeros(shape=(2*self.nz-1,2*self.ny-1,2*self.nx-1))
 
        self.padJx[0:self.nz, 0:self.ny, 0:self.nx] = Jx_
        self.padJy[0:self.nz, 0:self.ny, 0:self.nx] = Jy_
        self.padJz[0:self.nz, 0:self.ny, 0:self.nx] = Jz_
 
        if self.PBCx:
            self.padJx[0:self.nz, 0:self.ny, self.nx:] = Jx_[:,:,:-1]
            self.padJy[0:self.nz, 0:self.ny, self.nx:] = Jy_[:,:,:-1]
            self.padJz[0:self.nz, 0:self.ny, self.nx:] = Jz_[:,:,:-1]        
 
        if self.PBCy:
            self.padJx[0:self.nz, self.ny:, 0:self.nx] = Jx_[:,:-1,:]
            self.padJy[0:self.nz, self.ny:, 0:self.nx] = Jy_[:,:-1,:]
            self.padJz[0:self.nz, self.ny:, 0:self.nx] = Jz_[:,:-1,:]
 
        if self.PBCz:
            self.padJx[self.nz:, 0:self.ny, 0:self.nx] = Jx_[:-1,:,:]
            self.padJy[self.nz:, 0:self.ny, 0:self.nx] = Jy_[:-1,:,:]
            self.padJz[self.nz:, 0:self.ny, 0:self.nx] = Jz_[:-1,:,:]
 
        if self.PBCx & self.PBCy:
            self.padJx[0:self.nz, self.ny:, self.nx:] = Jx_[:,:-1,:-1]
            self.padJy[0:self.nz, self.ny:, self.nx:] = Jy_[:,:-1,:-1]
            self.padJz[0:self.nz, self.ny:, self.nx:] = Jz_[:,:-1,:-1]
 
        if self.PBCy & self.PBCz:
            self.padJx[self.nz:, self.ny:, 0:self.nx] = Jx_[:-1,:-1,:]
            self.padJy[self.nz:, self.ny:, 0:self.nx] = Jy_[:-1,:-1,:]
            self.padJz[self.nz:, self.ny:, 0:self.nx] = Jz_[:-1,:-1,:]
 
        if self.PBCz & self.PBCx:
            self.padJx[self.nz:, 0:self.ny, self.nx:] = Jx_[:-1,:,:-1]
            self.padJy[self.nz:, 0:self.ny, self.nx:] = Jy_[:-1,:,:-1]
            self.padJz[self.nz:, 0:self.ny, self.nx:] = Jz_[:-1,:,:-1]
 
        if self.PBCx & self.PBCy & self.PBCz:
            self.padJx[self.nz:, self.ny:, self.nx:] = Jx_[:-1,:-1,:-1]
            self.padJy[self.nz:, self.ny:, self.nx:] = Jy_[:-1,:-1,:-1]
            self.padJz[self.nz:, self.ny:, self.nx:] = Jz_[:-1,:-1,:-1]
 
        self.FpadJx = np.fft.fftn(self.padJx, axes=(0,1,2), norm=None)
        self.FpadJy = np.fft.fftn(self.padJy, axes=(0,1,2), norm=None)
        self.FpadJz = np.fft.fftn(self.padJz, axes=(0,1,2), norm=None)
 
        self.Hampx = np.real(np.fft.ifftn(                           + self.FKtotxy*self.FpadJy + self.FKtotxz*self.FpadJz, axes=(0,1,2), norm=None))
        self.Hampy = np.real(np.fft.ifftn(- self.FKtotxy*self.FpadJx                            + self.FKtotyz*self.FpadJz, axes=(0,1,2), norm=None))
        self.Hampz = np.real(np.fft.ifftn(- self.FKtotxz*self.FpadJx - self.FKtotyz*self.FpadJy                           , axes=(0,1,2), norm=None))
 
        self.hampx = self.Hampx[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]
        self.hampy = self.Hampy[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]
        self.hampz = self.Hampz[self.nz-1:2*self.nz,self.ny-1:2*self.ny,self.nx-1:2*self.nx]
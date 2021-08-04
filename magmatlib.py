import math

class material(object):
    def __init__(self):
        # Units
        self.units = {
            'Ms'  : 'emu/cc',
            'Aex' : 'erg/cm',
            'K1'  : 'erg/cc',
            'K2'  : 'erg/cc',
            'Ku'  : 'erg/cc',
            'thetaK' : 'deg',
            'phiK'  : 'deg',
            'gamma' : 'Oe^-1*s^-1',
            'alpha' : 'dimensionless',
            'DDMI' : 'erg/cm^2',
            'Temp' : 'K',
        }
            
    def properties(self):
        total = 50
        name = self.__class__.__name__
        side = total - len(name)
        l_side = side // 2 - 1
        r_side = l_side + side % 2
        print('='*l_side + ' ' + name + ' ' + '='*r_side)
        
        for k, v in self.__dict__.items():
            if k!='units':
                print('{} \t : {} {}'.format(k, v, self.units[k]))
                
        print('='*total)

class Ni_SingleCrystal(material):
    def __init__(self):
        super(Ni_SingleCrystal, self).__init__()
        
        # Magnetic properties
        self.Ms = 484            # [unit : emu/cc]
        self.Aex = 1.05*10**-6   # [unit : erg/cm]
        self.K1 = -5*10**4       # [unit : erg/cc]
        self.K2 = -2*10**4       # [unit : erg/cc]
        self.Ku = 0              # [unit : erg / cc]

        # Structural properties 
        self.thetaK = 0          # [unit : deg]
        self.phiK = 0            # [unit : deg]

        # Spin dynamics properties
        self.gamma = 1.76*10**7  # [unit : Oe^-1 s^-1]
        self.alpha = 10          # [unit : dimensionless]
        self.DDMI = 0            # [unit : erg / cm^2]

        # Thermal state
        self.Temp = 300          # [unit : K]
    
class Ni_PolyCrystal(material):
    def __init__(self):
        super(Ni_PolyCrystal, self).__init__()
        
        # Magnetic properties
        self.Ms = 484            # [unit : emu/cc]
        self.Aex = 1.05*10**-6   # [unit : erg/cm]
        self.K1 = 0              # [unit : erg/cc]
        self.K2 = 0              # [unit : erg/cc]
        self.Ku = 0              # [unit : erg / cc]

        # Structural properties 
        self.thetaK = 0          # [unit : deg]
        self.phiK = 0            # [unit : deg]

        # Spin dynamics properties
        self.gamma = 1.76*10**7  # [unit : Oe^-1 s^-1]
        self.alpha = 10          # [unit : dimensionless]
        self.DDMI = 0            # [unit : erg / cm^2]

        # Thermal state
        self.Temp = 300          # [unit : K]

class Ni_Preferred111PolyCrystal(material):
    def __init__(self):
        super(Ni_Preferred111PolyCrystal, self).__init__()
        
        # Magnetic properties
        self.Ms = 484            # [unit : emu/cc]
        self.Aex = 1.05*10**-6   # [unit : erg/cm]
        self.K1 = -5*10**4       # [unit : erg/cc]
        self.K2 = -2*10**4       # [unit : erg/cc]
        self.Ku = 0              # [unit : erg / cc]

        # Structural properties 
        self.thetaK = -math.atan(math.sqrt(2))*(180/math.pi)          # [unit : deg]
        self.phiK = 45           # [unit : deg]

        # Spin dynamics properties
        self.gamma = 1.76*10**7  # [unit : Oe^-1 s^-1]
        self.alpha = 10          # [unit : dimensionless]
        self.DDMI = 0            # [unit : erg / cm^2]

        # Thermal state
        self.Temp = 300          # [unit : K]
        
        
class CoFeB_Amorphous(material):
    def __init__(self):
        super(Ni_PolyCrystal, self).__init__()
        
        # Magnetic properties
        self.Ms = 1080            # [unit : emu/cc]
        self.Aex = 1.5*10**-6    # [unit : erg/cm]
        self.K1 = 0              # [unit : erg/cc]
        self.K2 = 0              # [unit : erg/cc]
        self.Ku = 8.0*10**6      # [unit : erg / cc]

        # Structural properties 
        self.thetaK = 0          # [unit : deg]
        self.phiK = 0            # [unit : deg]

        # Spin dynamics properties
        self.gamma = 1.76*10**7  # [unit : Oe^-1 s^-1]
        self.alpha = 0.1         # [unit : dimensionless]
        self.DDMI = 0            # [unit : erg / cm^2]

        # Thermal state
        self.Temp = 300          # [unit : K]
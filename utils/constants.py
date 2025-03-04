# utils/constants.py

# Physical constants
AVOGADRO_NUMBER = 6.022e23  # atoms/mol
EV_TO_JOULE = 1.602e-19  # J/eV
BARN_TO_CM2 = 1.0e-24  # cm^2/barn
MEV_PER_FISSION = 200  # MeV

# Default reactor parameters
DEFAULT_CORE_RADIUS = 190  # cm
DEFAULT_CORE_HEIGHT = 900  # cm
DEFAULT_FUEL_ENRICHMENT = 4.5  # %
DEFAULT_POWER = 80e6  # Wth (80 MWth)
DEFAULT_PACKING_FRACTION = 0.61  # Pebble packing fraction

# Neutronic parameters
DEFAULT_DIFFUSION_COEF = 1.18  # cm
DEFAULT_SIGMA_A = 0.0316  # cm^-1
DEFAULT_SIGMA_F = 0.0130  # cm^-1
DEFAULT_NU = 2.43  # Neutrons per fission

# Derived parameters
DEFAULT_L_SQUARED = DEFAULT_DIFFUSION_COEF / DEFAULT_SIGMA_A  # cm^2
DEFAULT_B_G_SQUARED = 1.73e-4  # cm^-2

__all__ = [
    # MOEAs
    "RVEA",
    "MOEAD",
    "NSGA2",
    "HypE",
    "RVEAa",
    "TensorMOEAD",
    "DNSGA2A",
    "DNSGA2B",
    "DC_MOEA",
    "dCMOEA",
    "TDCEA",
    "tensorDCEA"
]

from .mo import MOEAD, NSGA2, RVEA, HypE, RVEAa,  TensorMOEAD
from .DCMOEA import DNSGA2A, DNSGA2B, DC_MOEA, dCMOEA, TDCEA, tensorDCEA


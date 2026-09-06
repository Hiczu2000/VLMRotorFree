import numpy as np

from preprocessor.BladeSets import StructuralGeometry, AerodynamicGeometry, FixedVortexGrid
from utilities import Timer

def preprocessor(xlsx_path:str,
                 wing_mesh_density:tuple,
                 pitch_angle:float,
                 wind_velocity:float,
                 angular_velocity:float,
                 coordinate_system_origin:np.ndarray,
                 cut_off_thickness:float,
                 cosine_spacing:tuple = (False, False),
                 timestep:float = 0.1) -> tuple[FixedVortexGrid, AerodynamicGeometry]:
    """
    Input the modelling data and construct the 
    discretisized lifting surface representation based 
    them, furthermore create a fixed vortex grid.

    Parameters
    ----------
    xlsx_path : str
        Path to the .xlsx file with the blade definition.
    wing_mesh_density : tuple
        Number of (chordwise, spanwise) vortex rings, 
        representing the lifitng surface
    pitch_angle : float
        Pitch angle of the rotor [rad]
    wind_velocity : float
        Wind velocity [m/s]
    angular_velocity : float
        Angular velocity [rad/s]
    coordinate_system_origin : np.ndarray
        Center of the rotor [m]
    cut_off_thickness : float
        Cut off relative thickness [%]
    cosine_spacing : tuple, optional
        Enabling the cosine spacing in (chordwise, spanwise) directions, 
        by default (False, False)
    timestep : float, optional
        Fixed vortex grid timetep [s], 
        time after which the wake can 
        move freely,
        by default 0.1

    Returns
    -------
    FixedVortexGrid
        Representation of the fixed wake vortex grid
    AerodynamicGeometry
        Representation of the fixed blade vortex grid
    """
    
    with Timer("Structural geometry data imported in", shift_number = 1):
        structural_blade = StructuralGeometry(xlsx_path = xlsx_path,
                                              cut_off_thickness = cut_off_thickness) 
    
    with Timer("Aerodynamic geometry projected in", shift_number = 1):
        aerodynamic_blade = AerodynamicGeometry(structural_geometry = structural_blade,
                                                mesh_density = wing_mesh_density, 
                                                aoa = pitch_angle,
                                                cosine_spacing = cosine_spacing) 
  
    with Timer("Fixed vortex grid created in", shift_number = 1):      
        fixed_vortex_grid = FixedVortexGrid(aerodynamic_geometry = aerodynamic_blade,
                                            origin = coordinate_system_origin,
                                            wind_velocity = wind_velocity,
                                            angular_velocity = angular_velocity, 
                                            timestep = timestep)
        
        return fixed_vortex_grid, aerodynamic_blade
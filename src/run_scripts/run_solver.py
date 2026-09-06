from preprocessor.BladeSets import StructuralGeometry, AerodynamicGeometry, FixedVortexGrid
from utilities import Timer
from utilities import log, section, print_logo
from solver.biosavart import VortexCoreModels, KatzPlotkin, Scully, LambOseen
from solver import SolverUnsteadyVLM

def solver(fixed_vortex_grid:FixedVortexGrid,
           wind_velocity:float,
           air_density:float,
           angular_velocity:float,
           blade_number:int,
           simulation_name:str,
           timestep:float,
           number_of_timesteps:int,
           save_geometry_for_plot:bool = True,
           save_every:int = 1,
           viscous_drag_polars:list = None,
           cylinder_drag_coefficient:float = 0.6,
           extrapolate_drag_coefficients:bool = True,
           tolerance_of_drag_coefficient_extrapolation:float = 0.1,
           drag_approximation_verbose:bool = False,
           plot_drag_approximation:bool = False,
           averaged_drag_chordwise:bool = False,
           aero_geometry:AerodynamicGeometry = None,
           core_model: VortexCoreModels = None,
           core_model_free_wake:VortexCoreModels = None,
           core_model_verbose: bool = False):
    """
    Import the fixed vortex grid, initiate the free one
    and conduct the full computation procedure.

    Parameters
    ----------
    fixed_vortex_grid : FixedVortexGrid
        Representation of the fixed wake vortex grid
    wind_velocity : float
        Wind velocity [m/s]
    air_density : float
        Density of air [kg/m^3]
    angular_velocity : float
        Rotot angular velocity [rad/s]
    blade_number : int
        Number of rotor blades
    simulation_name : str
        Name of the simulation files
    timestep : float
        Free vortex grid timestep [s]
    number_of_timesteps : int
        Number of timesteps in simulation
    save_geometry_for_plot : bool, optional
        If True saves the vortex grid geometry
        to output files, 
        by default True
    save_every : int, optional
        The interval for which timestep 
        will save the .h5 results file,
        by default 1
    viscous_drag_polars : list, optional
        List of the polar names used for
        the viscous drag interpolation, 
        by default None
    cylinder_drag_coefficient : float, optional
        Cylinder drag coefficient used 
        as the maximum avaible drag 
        coefficient in the viscous drag
        interpolation, 
        by default 0.6
    extrapolate_drag_coefficients : bool, optional
        If True, extrapolates Cl/Cd beyond the defined data range using
        linear scipy.interp1d extrapolation. If False, values are clamped
        at the data boundary (numpy.interp behavior), by default True
    tolerance_of_drag_coefficient_extrapolation : float, optional
        Tolerance in finding an airfoil with
        the closest relative thickness to the 
        target one [%], by default 0.1
    drag_approximation_verbose : bool, optional
        Option of showing the logs about the 
        drag coefficient approximation, by default False        
    plot_drag_approximation : bool, optional
        Option of showing the plots of the
        drag coefficient approximation, by default False
    averaged_drag_chordwise : bool, optional
        If True, drag is computed from Cd * q * A and distributed
        equally across chordwise filaments. If False, drag is scaled
        per panel from lift using Cd/Cl ratio.
        Preferred to use for high pitch angles, 
        by default False
    aero_geometry : AerodynamicGeometry, optional
        Representation of the fixed blade vortex grid, 
        by default None
    core_model : VortexCoreModels, optional
        Vortex core model used for the fixed vortex rings, 
        by default None
    core_model_free_wake : VortexCoreModels, optional
        Vortex core model used for the free vortex rings, 
        by default None
    core_model_verbose : bool, optional
        Print the information about the used core models, 
        by default False

    Returns
    -------
    SolverUnsteadyVLM
        Solver containing the solution and settings
    """

    solver1 = SolverUnsteadyVLM(full_mesh = fixed_vortex_grid,
                                velocity = wind_velocity,
                                angular_velocity = angular_velocity,
                                blades = blade_number,
                                save_geometry_for_plot = save_geometry_for_plot)

    solver1.run(timestep = timestep,
                    number_of_timesteps = number_of_timesteps,
                    air_density = air_density,
                    simulation_name  = simulation_name,
                    save_every = save_every,
                    viscous_drag_polars = viscous_drag_polars,
                    cylinder_drag_coefficient = cylinder_drag_coefficient,
                    extrapolate_drag_coefficients = extrapolate_drag_coefficients,
                    tolerance_of_drag_coefficient_extrapolation = tolerance_of_drag_coefficient_extrapolation,
                    drag_approximation_verbose = drag_approximation_verbose,
                    plot_drag_approximation = plot_drag_approximation,
                    averaged_drag_chordwise = averaged_drag_chordwise,
                    aero_geometry = aero_geometry,
                    core_model = core_model,
                    core_model_free_wake = core_model_free_wake,
                    core_model_verbose = core_model_verbose
                    )
    return solver1
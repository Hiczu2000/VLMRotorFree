import numpy as np

from run_scripts import preprocessor, solver
from solver.biosavart import VortexCoreModels, KatzPlotkin, Scully, LambOseen
from utilities import log, section, print_logo, summary

# ---------------------------------------------------------------------------
# Run inputs 
# ---------------------------------------------------------------------------

# Rotor proporties
blade_number = 3                            # rotor blade number [-]
blade_input = 'blade_input/blade.xlsx'     # .xlsx file with defined blade properties
viscous_drag_polars = ['ffaw3241',          # viscous drag polars
                       'ffaw3301',
                       'ffaw3360',
                       'ffaw3480',
                       'ffaw3600',
                      ]
cylinder_drag_coefficient = 0.6             # [-]


# Operational properties (their list must have the same size)
V = [6, 8, 10, 12, 16, 20, 25]                                      # wind speed [m/s]
omega = np.array([6, 6.423, 8.029, 9.600, 9.600, 9.600, 9.600])     # rotor angular speed [rpm]
pitch = [0.789, 0.000, 0.000, 4.507, 12.042, 16.968, 22.175]        # blade pitch angle [rad]


# Physical properties
air_density = 1.225 #           [kg/m^3]
kinematic_viscosity = 1.5e-5 #  [m^2/s]
lambs_constant = 1.25643 #      [-]


# Discretization properties
tc = 80                                             # cut off relative thickness [%]
spanwise_mesh = 24                                  # vortex ring number in spanwise direction [-]
chordwise_mesh = 10                                 # vortex ring number in chordwise direction [-]
fixed_timestep = 0.05                               # fixed vortex grid timestep [s]
dt = 0.4                                            # free vortex grid timestep [s]
simulation_times = [100, 84, 84, 60, 30, 30, 30]     # simulation lengths [s] (must have the same size as operational properties)


# Vortex model properties
core_ratio = np.array([0.0014])     # ratio of the intial vortex core radius to the local chord [-]
a1_list = np.array([0.0001])        # spanwise diffusion rate [-]


# Define the order of operational points to run
run_order = [5]
simulation_name = 'my_run'

# ---------------------------------------------------------------------------
# Unchangable code
# ---------------------------------------------------------------------------

print_logo("VLMRotorFree")    

for cr in core_ratio:
    for a1 in a1_list:
        for idx in run_order:

            v = V[idx]
            rot = omega[idx]
            p = pitch[idx]
            st = simulation_times[idx]

            log(f"Wind speed: {v} m/s",color = True)
            log(f"Simulation time: {st:.2f} s",color = True)
            
            n_steps = int(round(st / dt)) + 1

            name = f"{simulation_name}_wsp{v}_mesh{chordwise_mesh}_{spanwise_mesh}_tc{tc}_dt{dt}_rc{cr}_a1{a1}"

            log(f"Running {name}: dt = {dt}, steps = {n_steps - 1}",color = True)

            section("Preprocessor")
            log("Geometry computation...", "STEP")

            meshgrid, aerodynamic_blade = preprocessor(
                xlsx_path= blade_input,
                wing_mesh_density=(chordwise_mesh, spanwise_mesh),
                pitch_angle=-np.deg2rad(p),
                wind_velocity=v,
                angular_velocity=rot * np.pi / 30,
                coordinate_system_origin=np.array([0, 0, 0]),
                cut_off_thickness=tc,
                cosine_spacing=(True, True),
                timestep=fixed_timestep
            )

            log("Geometry created successfully", "OK", color = True)

            free_wake_initial_time = np.ones(aerodynamic_blade.n_spanwise - 1) * (fixed_timestep + (dt * 0.5)) 

            solution = solver(
                fixed_vortex_grid=meshgrid,
                wind_velocity=v,
                air_density=air_density,
                angular_velocity=rot * np.pi / 30,
                blade_number=blade_number,
                save_geometry_for_plot=True,
                simulation_name=name,
                timestep=dt,
                number_of_timesteps=n_steps,
                save_every=1,
                viscous_drag_polars= viscous_drag_polars,
                cylinder_drag_coefficient = cylinder_drag_coefficient,
                extrapolate_drag_coefficients=True,
                tolerance_of_drag_coefficient_extrapolation=0.1,
                drag_approximation_verbose=False,
                plot_drag_approximation=False,
                averaged_drag_chordwise=True,
                aero_geometry=aerodynamic_blade,
                core_model= Scully(initial_core_size = meshgrid.vortex_local_chords * cr),
                core_model_free_wake = Scully(initial_core_size = aerodynamic_blade.between_section_chords * cr,
                                              vortex_diffusion_times = free_wake_initial_time,
                                              lambs_constant = lambs_constant ,
                                              a1 = a1,
                                              kinematic_viscosity = kinematic_viscosity),
                core_model_verbose=False
            )
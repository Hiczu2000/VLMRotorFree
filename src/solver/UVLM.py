import numpy as np
import pandas as pd
import time

from preprocessor import VectorSet, FixedVortexGrid, AerodynamicGeometry, Mesh2D
from solver.CoefficientMatrix import ComputationPanels, FreeVortexGrid
from utilities import log, section, Timer, summary
from solver.biosavart import VortexCoreModels, Scully, LaminarVortexCoreModels
from pathlib import Path

class SolverUnsteadyVLM:
    """Unsteady free-wake Vortex Lattice Method solver for rotor blade
    aerodynamics, including free wake convection, lift via Kutta-Joukowski,
    and optional viscous drag correction from polar data.
    """
    def __init__(self, 
                 full_mesh:FixedVortexGrid,
                 velocity:float, 
                 angular_velocity:float,
                 blades:int = 3,
                 save_geometry_for_plot:bool = False):
        """
        Initiate the unsteady vortex lattice method solver

        Parameters
        ----------
        full_mesh : FixedVortexGrid
            Fixed Vortex grid contains the vortex rings from 
            the lifting surface and the first spanwise 
            row of the wake
        velocity : float
            Free stream velocity in axial direction
        angular_velocity : float
            Free stream velocity in tangent direction
        blades : int, optional
            Blade number in rotor, by default 3
        save_geometry_for_plot : bool, optional
            Save computational grid geometry in the result file 
            in order to visualize it later, by default False
        """
    
        self.full_mesh = full_mesh
        self.computation_panels = ComputationPanels(full_mesh = full_mesh)
        self.circulation_vectors =  full_mesh.gamma_vectors
        self.V = velocity
        self.omega = angular_velocity
        self.blades = blades

        self.trailing_edge = full_mesh.trailin_edge
        self.save_geometry_for_plot = save_geometry_for_plot
        
        log("Solver set up correctly", level = "OK", color = True)

        summary("Solver settings", {
                "Wind velocity [m/s]": self.V,
                "Angular velocity [rad/s]": self.omega,
                "Blade number":  self.blades,
                "Mesh size (one blade)": self.computation_panels.vortex_grid.X.shape,
            })


    def run(self,
            timestep:float,
            number_of_timesteps:int,
            air_density:float = 1.225,
            simulation_name:str = 'name',
            save_every:int = 1,
            viscous_drag_polars:list = None,
            cylinder_drag_coefficient:float = 0.6,
            extrapolate_drag_coefficients:bool = True,
            tolerance_of_drag_coefficient_extrapolation:float = 0.1,
            drag_approximation_verbose:bool = False,
            plot_drag_approximation:bool = False,
            averaged_drag_chordwise:bool = False,
            aero_geometry:AerodynamicGeometry = None,
            core_model:VortexCoreModels = None,
            core_model_free_wake:VortexCoreModels = None,
            core_model_verbose:bool = False
            ):
        """
        Run the unsteady free wake simulation

        Parameters
        ----------
        timestep : float
            Size of the timestep
        number_of_timesteps : int
            Nuber of the timesteps used in the simulation
        air_density : float, optional
            Air density used for the lift computation, by default 1.225
        simulation_name : str, optional
            Name of the simulation, by default 'name'
        save_every : int, optional
            For each timestep the results are saved, by default 1
        viscous_drag_polars : list, optional
            Name of the polars used for the viscous drag interpolation,
            by default None
        cylinder_drag_coefficient : float, optional
            The maximum drag coefficient used in the viscous drag interpolation,
            by default 0.6
        extrapolate_drag_coefficients : bool, optional
            Extrapolate the polars beyond the data in the polars, by default True
        tolerance_of_drag_coefficient_extrapolation : float, optional
            Margin with which the coefficients are interpolated, by default 0.1
        drag_approximation_verbose : bool, optional
            Print the information about the drag approximation, by default False
        plot_drag_approximation : bool, optional
            Plot the information about the drag approximation, by default False
        averaged_drag_chordwise : bool, optional
            Average the viscous drag per spanwise section, by default False
        aero_geometry : AerodynamicGeometry, optional
            Aerodynamic discretization of the lifting surface,
            required for the viscous drag computation,
            by default None
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        core_model_free_wake : VortexCoreModels, optional
            Vortex core model used for the free vortex rings, by default None
        core_model_verbose : bool, optional
            Print the information about the used core models, by default False

        Raises
        ------
        FileExistsError
            If the simulation with the same name already exists,
            the simulation will not start
        """
        
        simulation_folder = Path("results") / simulation_name

        if simulation_folder.exists():
            raise FileExistsError(
                f"Folder {simulation_folder} already exists. Give other name to not overwrite the results."
                )
        else:
            simulation_folder.mkdir(parents = True)

        # zero timestep
        timestep_number = 0

        fixed_grid = self.computation_panels.vortex_grid

        free_wake_start_line = Mesh2D(x_set = fixed_grid.X[-1,:],
                                      y_set = fixed_grid.Y[-1,:],
                                      z_set = fixed_grid.Z[-1,:])

        free_wake = FreeVortexGrid(start_line = free_wake_start_line) 
        
        timing_log = {
                    'timestep': [],
                    'time_sim': [],
                    'time_total': [],
                    'time_wake_movement_computation': [],
                    'time_gamma_computation': [],
                    'time_lift_computation': [],
                    'time_drag_computation': []
        }

        while timestep_number <= number_of_timesteps - 1:
            timestep_number += 1

            # neglecting the fixed vortex timestep
            physical_time = (timestep_number - 1) * timestep

            section(f'timestep: {timestep_number}, time: {physical_time}')
            t_step_start = time.perf_counter()

            # 1st timestep
            if timestep_number == 1: 
                
                t_wake_movement = 0


                t0 = time.perf_counter()
                # free wake does not exist in the first timestep
                self.compute_gamma(free_wake_contribution = None,
                                   save_velocity_vector = True,
                                   core_model = core_model,
                                   core_model_verbose = core_model_verbose)
                t_gamma = time.perf_counter() - t0
                
                if timestep_number % save_every == 0:

                    # forces computed only for the saved results
                    t0 = time.perf_counter()
                    # free wake does not exist in the first timestep
                    self.compute_lift(air_density = air_density,
                                      save_velocity_vector = True,
                                      include_induced_velocities = True,
                                      free_wake_contribution = None,
                                      core_model = core_model,
                                      core_model_verbose = core_model_verbose)
                    t_lift = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if viscous_drag_polars is not None:
                        self.compute_viscous_drag(viscous_drag_polars = viscous_drag_polars,
                                                  chords_of_section = aero_geometry.between_section_chords,
                                                  relative_thicknesses_of_section = aero_geometry.relative_thickness_of_between_section_chords,
                                                  cylinder_drag_coefficient = cylinder_drag_coefficient,
                                                  extrapolate = extrapolate_drag_coefficients,
                                                  tol = tolerance_of_drag_coefficient_extrapolation,
                                                  verbose = drag_approximation_verbose,
                                                  plot_interpolation = plot_drag_approximation,
                                                  averaged_drag_chordwise = averaged_drag_chordwise
                                                )
                    t_drag = time.perf_counter() - t0

            # second timestep
            elif timestep_number ==2: 
                
                t0 = time.perf_counter()
                # induced velocities are used to compute the wake convection, 
                # since free  vortex is only a geometric line, it does not 
                # contribute to the convection of itself
                U, V, W  = self.induced_velocities(target_points = free_wake.vortex_grid,
                                                   wing_symmetry = False,
                                                   free_wake_contribution = None,
                                                   core_model = core_model,
                                                   core_model_verbose = core_model_verbose
                                                   )

                grid_shape = free_wake.vortex_grid.X.shape

                wake_induced_velocities = Mesh2D(x_set = U.reshape(grid_shape),
                                                 y_set = V.reshape(grid_shape),
                                                 z_set = W.reshape(grid_shape))
                
                free_wake.update_position(induced_velocities = wake_induced_velocities,
                                          angular_velocity = self.omega,
                                          wind_velocity = self.V,
                                          timestep = timestep) 

                t_wake_movement = time.perf_counter() - t0

                free_wake.update_gamma(new_gamma_line = self.gamma[-self.computation_panels.nbs:])

                # vortex diffusion initialization
                if core_model_free_wake is not None and getattr(core_model_free_wake, "update", False):
                    core_model_free_wake.initiate_diffusion(gamma = free_wake.gamma)

                t0 = time.perf_counter()

                # after the wake convection the gamma computation has the 
                # free wake contribution 
                self.compute_gamma(free_wake_contribution = free_wake,
                                   save_velocity_vector = True,
                                   core_model = core_model,
                                   core_model_free_wake = core_model_free_wake,
                                   core_model_verbose = core_model_verbose)

                t_gamma = time.perf_counter() - t0
                
                if timestep_number % save_every == 0:
                        
                    t0 = time.perf_counter()                
                    self.compute_lift(air_density = air_density,
                                      save_velocity_vector = True,
                                      include_induced_velocities = True,
                                      free_wake_contribution = free_wake,
                                      core_model = core_model,
                                      core_model_free_wake = core_model_free_wake,
                                      core_model_verbose = core_model_verbose)
                    t_lift = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if viscous_drag_polars is not None:
                        self.compute_viscous_drag(viscous_drag_polars = viscous_drag_polars,
                                                  chords_of_section = aero_geometry.between_section_chords,
                                                  relative_thicknesses_of_section = aero_geometry.relative_thickness_of_between_section_chords,
                                                  cylinder_drag_coefficient = cylinder_drag_coefficient,
                                                  extrapolate = extrapolate_drag_coefficients,
                                                  tol = tolerance_of_drag_coefficient_extrapolation,
                                                  verbose = drag_approximation_verbose,
                                                  plot_interpolation = plot_drag_approximation,
                                                  averaged_drag_chordwise = averaged_drag_chordwise
                                                  )
                    t_drag = time.perf_counter() - t0

            # Next steps
            else:
                t0 = time.perf_counter()

                # After the creating of free wake vortex rings, the computed 
                # induced velocities have free wake contribution
                U, V, W  = self.induced_velocities(target_points = free_wake.vortex_grid,
                                                   wing_symmetry = False,
                                                   free_wake_contribution = free_wake,
                                                   core_model = core_model,
                                                   core_model_free_wake = core_model_free_wake,
                                                   core_model_verbose = core_model_verbose
                                                   )
                
                grid_shape = free_wake.vortex_grid.X.shape

                wake_induced_velocities = Mesh2D(x_set = U.reshape(grid_shape),
                                                 y_set = V.reshape(grid_shape),
                                                 z_set = W.reshape(grid_shape))
                
                free_wake.update_position(induced_velocities = wake_induced_velocities,
                                          angular_velocity = self.omega,
                                          wind_velocity = self.V,
                                          timestep = timestep)
                t_wake_movement = time.perf_counter() - t0

                free_wake.update_gamma(new_gamma_line = self.gamma[-self.computation_panels.nbs:])

                if core_model_free_wake is not None:
                    core_model_free_wake.update_model_in_time(timestep=timestep,
                                                              n_spanwise=self.computation_panels.nbs,
                                                              gamma = free_wake.gamma)

                t0 = time.perf_counter()
                self.compute_gamma(free_wake_contribution = free_wake,
                                   save_velocity_vector = True,
                                   core_model = core_model,
                                   core_model_free_wake = core_model_free_wake,
                                   core_model_verbose = core_model_verbose
                                   )
                t_gamma = time.perf_counter() - t0

                if timestep_number % save_every == 0:

                    t0 = time.perf_counter()
                    self.compute_lift(air_density = air_density,
                                      save_velocity_vector = True,
                                      include_induced_velocities = True,
                                      free_wake_contribution = free_wake,
                                      core_model = core_model,
                                      core_model_free_wake = core_model_free_wake,
                                      core_model_verbose = core_model_verbose
                                      )
                    t_lift = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    if viscous_drag_polars is not None:
                        self.compute_viscous_drag(viscous_drag_polars = viscous_drag_polars,
                                                  chords_of_section = aero_geometry.between_section_chords,
                                                  relative_thicknesses_of_section = aero_geometry.relative_thickness_of_between_section_chords,
                                                  cylinder_drag_coefficient = cylinder_drag_coefficient,
                                                  extrapolate = extrapolate_drag_coefficients,
                                                  tol = tolerance_of_drag_coefficient_extrapolation,
                                                  verbose = drag_approximation_verbose,
                                                  plot_interpolation = plot_drag_approximation,
                                                  averaged_drag_chordwise = averaged_drag_chordwise
                                                  )
                    t_drag = time.perf_counter() - t0
                    
            t_step_total = time.perf_counter() - t_step_start

            if timestep_number % save_every == 0:

                timing_log['timestep'].append(timestep_number)
                timing_log['time_sim'].append(round(physical_time, 4))
                timing_log['time_total'].append(round(t_step_total, 4))
                timing_log['time_wake_movement_computation'].append(round(t_wake_movement, 4))
                timing_log['time_gamma_computation'].append(round(t_gamma, 4))
                timing_log['time_lift_computation'].append(round(t_lift, 4))
                timing_log['time_drag_computation'].append(round(t_drag, 4))

            t_str = f"{physical_time:.3f}".replace('.', '_')
            if timestep_number != 1:
                self.free_wake_gamma = free_wake.gamma
            if timestep_number % save_every == 0:
                self.save_results(file_name = simulation_folder / f'{simulation_name}_{timestep_number}_t{t_str}.h5')
        
        timing_df = pd.DataFrame(timing_log)
        timing_df.to_csv(simulation_folder/f'{simulation_name}_computation_time.csv', index = False)
        log(f"Timing report saved to {simulation_folder / f'{simulation_name}_computation_time.csv'}", level="OK", color=True)
        
        self.save_results(file_name = simulation_folder / f'{simulation_name}_final_t{t_str}.h5')


    def compute_gamma(self, 
                      free_wake_contribution:FreeVortexGrid = None,
                      save_velocity_vector:bool = True,
                      core_model:VortexCoreModels = None,
                      core_model_free_wake:VortexCoreModels = None,
                      core_model_verbose:bool = False):
        """
        Vortex filament strength computation

        Parameters
        ----------
        free_wake_contribution : FreeVortexGrid, optional
            Influence contribution from the free vortex grid,
            attributed to the right hand side of the solving equation, 
            by default None
        save_velocity_vector : bool, optional
            Saving in result file the velocities in
            collocation points, 
            by default True
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        core_model_free_wake : VortexCoreModels, optional
            Vortex core model used for the free vortex rings, by default None
        core_model_verbose : bool, optional
            Print the information about the used core models, by default False
        """

        nx = self.computation_panels.normals.X.ravel()[:, np.newaxis]
        ny = self.computation_panels.normals.Y.ravel()[:, np.newaxis]
        nz = self.computation_panels.normals.Z.ravel()[:, np.newaxis]
            
        # original cs:  x - chordwise, y - spanwise
        # rotor cs:    x - in plane,  y - bladewise, z - out of plane

        x = self.computation_panels.collocation_points.X.ravel()
        y = self.computation_panels.collocation_points.Y.ravel()

        omega = self.omega
        
        V_x = omega * y
        V_y = -omega * x

        RHS = -(V_x[:, np.newaxis] * nx + V_y[:, np.newaxis] * ny + self.V * nz)
        
        if free_wake_contribution is not None:
            A_free = self.computation_panels.compute_influence_matrix(wing_symmetry = False,
                                                                      blades = self.blades,
                                                                      save_geometry_for_plot = True,
                                                                      different_vortex_source = free_wake_contribution,
                                                                      core_model = core_model_free_wake,
                                                                      verbose = core_model_verbose)
            
            free_wake_induction = A_free @ free_wake_contribution.gamma
            RHS = RHS - free_wake_induction
        
        shape = V_x.shape
        velocities_in_collocation_points = VectorSet(x_set = V_x,
                                                     y_set = V_y,
                                                     z_set = self.V * np.ones(shape))

        if save_velocity_vector:
            self.V_in_cp = velocities_in_collocation_points

        log("AIC matrix computation...", "STEP")
        with Timer("AIC matrix computed in"):
            A = self.computation_panels.compute_influence_matrix(wing_symmetry = False,
                                                                 blades = self.blades,
                                                                 save_geometry_for_plot = self.save_geometry_for_plot,
                                                                 core_model = core_model,
                                                                 verbose = core_model_verbose)
        
        log("AIC matrix computed", "OK", color = True)
        
        log("Gamma computation...", "STEP")
        with Timer("Gamma computed in"):
            A_inv = np.linalg.inv(A)
            self.gamma = A_inv @ RHS


    def induced_velocities(self,
                           target_points:Mesh2D,
                           wing_symmetry:bool = False,
                           free_wake_contribution:FreeVortexGrid = None,
                           core_model:VortexCoreModels = None,
                           core_model_free_wake:VortexCoreModels = None,
                           core_model_verbose:bool = False) -> np.ndarray:
        """
        Compute induced velocites of the set up 
        on arbitrary target points

        Parameters
        ----------
        target_points : Mesh2D
            The locations of evaluating the induced velocity
        wing_symmetry : bool, optional
            Symmetrical vortex grid computation,
            used for the wing symmetry, 
            by default False
        free_wake_contribution : FreeVortexGrid, optional
            Influence contribution from the free vortex grid, 
            by default None
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        core_model_free_wake : VortexCoreModels, optional
            Vortex core model used for the free vortex rings, by default None
        core_model_verbose : bool, optional
            Print the information about the used core models, by default False

        Returns
        -------
        np.ndarray
            Velocities induced in X direction
        np.ndarray
            Velocities induced in Y direction
        np.ndarray
            Velocities induced in Z direction
        """
    
        if hasattr(self, 'gamma'):
            log("Induced velocities computation...", "STEP")
            with Timer("Induced velocities computed in"):
                U, V, W, = self.computation_panels.induced_velocities(target_points = target_points,
                                                                      gamma = self.gamma,
                                                                      wing_symmetry = wing_symmetry,
                                                                      blades = self.blades,
                                                                      core_model = core_model,
                                                                      verbose = core_model_verbose)
                if free_wake_contribution is not None:
                    U_free, V_free, W_free = free_wake_contribution.induced_velocities(gamma = free_wake_contribution.gamma,
                                                                                       target_points = target_points,
                                                                                       wing_symmetry = wing_symmetry,
                                                                                       blades = self.blades,
                                                                                       core_model = core_model_free_wake,
                                                                                       verbose = core_model_verbose)
                    U = U + U_free
                    V = V + V_free
                    W = W + W_free

            return U, V, W
        else:
            raise ValueError('Gamma is not computed yet, use compute_gamma before')   


    def compute_lift(self, 
                     air_density:float = 1.225,
                     save_velocity_vector:bool = False,
                     include_induced_velocities:bool = False,
                     free_wake_contribution:FreeVortexGrid = None,
                     core_model:VortexCoreModels = None,
                     core_model_free_wake:VortexCoreModels = None,
                     core_model_verbose:bool = False):
        """
        Compute lift forces, by using the vectorial form
        of the Kutta - Joukowski theorem

        Parameters
        ----------
        air_density : float, optional
            The density of air [kg/m^3], by default 1.225
        save_velocity_vector : bool, optional
            Saving in result file the velocities 
            used in the Kutta - Joukowiski computation, 
            by default True
        include_induced_velocities : bool, optional
            Include the induced velocities contribution
            in the Kutta - Joukowski equation, 
            by default False
        free_wake_contribution : FreeVortexGrid, optional
            Influence contribution from the free vortex grid,
            by default None
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        core_model_free_wake : VortexCoreModels, optional
            Vortex core model used for the free vortex rings, by default None
        core_model_verbose : bool, optional
            Print the information about the used core models, by default False
        """
      
        nbs = self.computation_panels.nbs

        circulation_by_panels = self.gamma.reshape(-1,nbs) 
        circulation_by_filaments = np.zeros_like(circulation_by_panels)
        circulation_by_filaments[0,:] = circulation_by_panels[0,:]
        circulation_by_filaments[1:,:] = circulation_by_panels[1:,:] - circulation_by_panels[:-1,:]

        blade_meshes = []

        blade_chordwise_size = self.computation_panels.collocation_points.shape[0] + 1
        entire_mesh = self.computation_panels.entire_mesh
        for mesh in entire_mesh:
            blade_mesh, wake_mesh = mesh.split_set_by_row(row_number = blade_chordwise_size)
            blade_meshes.append(blade_mesh)
        
        blade_mesh = blade_meshes[0]
        
        x = blade_mesh.spanwise_midpoints.X[:-1,:]
        y = blade_mesh.spanwise_midpoints.Y[:-1,:]

        omega = self.omega

        V_x = omega * y
        V_y = -omega * x

        shape = V_x.shape

        U_x = np.zeros(shape)
        U_y = np.zeros(shape)
        U_z = np.zeros(shape)

        if include_induced_velocities:
            lift_locations = Mesh2D(x_set = x,
                                    y_set = y,
                                    z_set = blade_mesh.spanwise_midpoints.Z[:-1,:])
            
            if free_wake_contribution is None:
                U_x, U_y, U_z = self.induced_velocities(target_points = lift_locations,
                                                        core_model = core_model,
                                                        core_model_verbose = core_model_verbose)
            else:
                U_x, U_y, U_z = self.induced_velocities(target_points = lift_locations,
                                                        free_wake_contribution = free_wake_contribution,
                                                        core_model = core_model,
                                                        core_model_free_wake = core_model_free_wake,
                                                        core_model_verbose = core_model_verbose)
            
            U_x = U_x.reshape(-1,nbs)
            U_y = U_y.reshape(-1,nbs)
            U_z = U_z.reshape(-1,nbs)

            induced_velocities = VectorSet(x_set = U_x,
                                           y_set = U_y,
                                           z_set = U_z)

            if save_velocity_vector:
                self.induced_velocities_in_lift_location = induced_velocities

        velocities_in_lift_location = VectorSet(x_set = V_x + U_x,
                                                y_set = V_y + U_y,
                                                z_set = self.V * np.ones(shape) + U_z)

        if save_velocity_vector:
            self.V_in_lift_location = velocities_in_lift_location

        self.lifts = (velocities_in_lift_location.cross_with_set(other_set = self.circulation_vectors) * 
                     (air_density * circulation_by_filaments))
    
        self.air_density = air_density

        self.lifts_per_section = VectorSet(blank = True,
                                           size = nbs)
        self.lifts_per_section.X = np.sum(self.lifts.X, axis = 0)
        self.lifts_per_section.Y = np.sum(self.lifts.Y, axis = 0)
        self.lifts_per_section.Z = np.sum(self.lifts.Z, axis = 0)


    def compute_viscous_drag(self,
                             viscous_drag_polars:list,
                             chords_of_section:np.array,
                             relative_thicknesses_of_section:np.array,
                             cylinder_drag_coefficient:float = 0.6,
                             extrapolate:bool = True,
                             tol:float = 0.1,
                             verbose:bool = False,
                             plot_interpolation:bool = False,
                             averaged_drag_chordwise:bool = False):
        """
        Compute viscou drag forces for the
        obtained lift forces based on the 
        aerodynamic coefficient interpolation
        by using the external polar files.

        Parameters
        ----------
        viscous_drag_polars : list
            List of the airfoil names
            related to the polar 
            cl_cd_[list_element].dat 
            files existing in the 
            blade_input folder
        chords_of_section : np.array
            Representative chord length
            of every spanwise section [m]
        relative_thicknesses_of_section : np.array
            Relative thickness of the 
            considered profiles [%]
        cylinder_drag_coefficient : float, optional
            Drag coefficient used for the interpolation 
            of profiles with relative thickness above
            the highest defined one in the airfoil set, 
            by default 0.6
        extrapolate : bool, optional
            Option of an extrapolation the Cl/Cd out of
            the defined Cl data by the linear scipy.interp1d
            extrapolation, when False (default), values are 
            clamped at the data boundary using numpy.interp
        tol : float, optional
            Tolerance in finding an airfoil with
            the closest relative thickness to the 
            target one [%], by default 0.1
        verbose : bool, optional
            Option of showing the logs about the 
            drag coefficient approximation, by default False
        plot_interpolation : bool, optional
            Option of showing the plots of the
            drag coefficient approximation, by default False
        averaged_drag_chordwise : bool, optional
            Apply the average in chordwise direction
            for every spanwise section, 
            it decreases the resolution of the viscous
            drag force, it's helpful to avoid numerical
            errors for sections with strong lift forces
            acting in both direction,
            by default False
        """

        if hasattr(self, 'lifts_per_section') and hasattr(self, 'V_in_lift_location'):
            from solver.viscous_drag import DragCoefficientInterpolator, viscous_drag

            interpolator = DragCoefficientInterpolator(airfoil_names = viscous_drag_polars,
                                                       cylinder_drag_coefficient = cylinder_drag_coefficient)
            
            lift_magnitudes_per_section = self.lifts_per_section.norms / self.full_mesh.panel_widths 
            velocity_magnitudes_in_lift_location = self.V_in_lift_location.norms
            velocity_magnitudes_per_section = np.mean(velocity_magnitudes_in_lift_location, axis = 0)

            dynamic_pressure_times_area =  0.5 * self.air_density * chords_of_section * velocity_magnitudes_per_section ** 2
            lift_coefficients_per_section = lift_magnitudes_per_section / dynamic_pressure_times_area 
            drag_coefficients_per_section = np.zeros_like(lift_coefficients_per_section)
            
            for i, (cl, rt) in enumerate(zip(lift_coefficients_per_section,
                                 relative_thicknesses_of_section)):
                cd = interpolator.interpolate(target_thickness = rt,
                                              target_lift_coefficient = cl,
                                              extrapolate = extrapolate,
                                              tol = tol,
                                              verbose = verbose,
                                              plot_interpolation = plot_interpolation)
                
                drag_coefficients_per_section[i] = cd
                
            drags = viscous_drag(lift_magnitudes = self.lifts.norms,
                                 cl_per_section = lift_coefficients_per_section,
                                 cd_per_section = drag_coefficients_per_section,
                                 drag_directions = self.V_in_lift_location.normalized,
                                 averaged_drag_chordwise = averaged_drag_chordwise,
                                 dynamic_pressure_times_area = dynamic_pressure_times_area,
                                 panel_widths = self.full_mesh.panel_widths)
            
            self.drags = drags

            nbs = self.computation_panels.nbs
            self.drags_per_section = VectorSet(blank = True,
                                               size = nbs)
            self.drags_per_section.X = np.sum(self.drags.X, axis = 0)
            self.drags_per_section.Y = np.sum(self.drags.Y, axis = 0)
            self.drags_per_section.Z = np.sum(self.drags.Z, axis = 0)

        else:
            if hasattr(self, 'lifts_per_section') is False:
                raise ValueError('The lifts per section needs to be computed firstly, drag computation is based on the Cl/Cd polars')
            if hasattr(self, 'V_in_lift_location') is False:
                raise ValueError('The velocities in lift location needs to be saved during the lift computation')


    def save_results(self, file_name:str, compression:str = 'lzf'):
        """
        Save all important data found
        in the solver to the external 
        .h5 file.

        Parameters
        ----------
        file_name : str
            Result file name
        compression : str, optional
            Required type of .h5
            file compression, 
            by default 'lzf'
        """
        import h5py

        with Timer("Results saved in"):
            with h5py.File(file_name, 'w') as f:
                

                f.attrs['wind_velocity'] = self.V
                f.attrs['blade_chordwise_mesh_elements'] = self.computation_panels.collocation_points.shape[0] + 1
                f.attrs['blade_spanwise_mesh_elements'] = self.computation_panels.collocation_points.shape[1] + 1
                f.attrs['panel_widths'] = self.full_mesh.panel_widths
                f.attrs['cosine_spacing'] = self.full_mesh.cosine_spacing

             
                f.attrs['omega'] = self.omega
                f.attrs['number_of_blades'] = self.blades

                geometry = f.create_group('geometry')

                mesh = self.computation_panels.entire_mesh

                panel_normals = geometry.create_group('panel_normals')
                panel_normals.create_dataset('X',
                                             data = self.computation_panels.normals.X,
                                             compression = compression)
                panel_normals.create_dataset('Y',
                                             data = self.computation_panels.normals.Y,
                                             compression = compression)
                panel_normals.create_dataset('Z',
                                             data = self.computation_panels.normals.Z,
                                             compression = compression)
                
                gamma_vec = geometry.create_group('gamma_vec')
                gamma_vec.create_dataset('X',
                                         data = self.circulation_vectors.X,
                                         compression = compression)
                gamma_vec.create_dataset('Y',
                                         data = self.circulation_vectors.Y,
                                         compression = compression)
                gamma_vec.create_dataset('Z',
                                         data = self.circulation_vectors.Z,
                                         compression = compression)
                
                if hasattr(self, 'trailing_edge'):
                    trailing_edge = geometry.create_group('trailing_edge')
                    trailing_edge.create_dataset('X',
                                                 data = self.trailing_edge.X,
                                                 compression = compression)                    
                    trailing_edge.create_dataset('Y',
                                                 data = self.trailing_edge.Y,
                                                 compression = compression)
                    trailing_edge.create_dataset('Z',
                                                 data = self.trailing_edge.Z,
                                                 compression = compression)


                for i in range(self.blades):
                    blade_group = geometry.create_group(f'blade_{i+1}')

                    blade_group.create_dataset('X',
                                                data = mesh[i].X,
                                                compression = compression)
                    blade_group.create_dataset('Y',
                                                data = mesh[i].Y,
                                                compression = compression)
                    blade_group.create_dataset('Z',
                                                data = mesh[i].Z,
                                                compression = compression)      

                if hasattr(self.computation_panels, 'free_wake_mesh'):
                    free_mesh = self.computation_panels.free_wake_mesh
                    for i in range(self.blades):
                        free_group = geometry.create_group(f'free_wake_{i+1}')
                    
                        free_group.create_dataset('X',
                                                    data = free_mesh[i].X,
                                                    compression = compression)
                        free_group.create_dataset('Y',
                                                    data = free_mesh[i].Y,
                                                    compression = compression)
                        free_group.create_dataset('Z',
                                                    data = free_mesh[i].Z,
                                                    compression = compression)      

                log('Geometry saved', level = "OK", color = True)

                if hasattr(self, 'gamma'):
                    gamma = f.create_group('gamma')
                    gamma.create_dataset('gamma',
                                         data = self.gamma,
                                         compression = compression)
                
                    log('Gammas saved', level = "OK", color = True)

                
                if hasattr(self, 'free_wake_gamma'):
                    fw_gamma = f.create_group('fw_gamma')
                    fw_gamma.create_dataset('fw_gamma',
                                         data = self.free_wake_gamma,
                                         compression = compression)
                
                    log('Gammas from the free wake saved', level = "OK", color = True)


                if hasattr(self, 'lifts'):
                    lifts = f.create_group('lifts')

                    lifts.create_dataset('X',
                                         data = self.lifts.X,
                                         compression = compression)
                    lifts.create_dataset('Y',
                                         data = self.lifts.Y,
                                         compression = compression)
                    lifts.create_dataset('Z',
                                         data = self.lifts.Z,
                                         compression = compression)  
                    f.attrs['air_density'] = self.air_density
                    log('Lift forces saved', level = "OK", color = True)

                if hasattr(self, 'moments'):
                    moments = f.create_group('moments')

                    moments.create_dataset('X',
                                           data = self.moments.X,
                                           compression = compression)
                    moments.create_dataset('Y',
                                           data = self.moments.Y,
                                           compression = compression)
                    moments.create_dataset('Z',
                                           data = self.moments.Z,
                                           compression = compression)

                    log('Moments saved', level = "OK", color = True)


                if hasattr(self, 'moments_per_section'):
                    moments_per_section = f.create_group('moments_per_section')

                    moments_per_section.create_dataset('X',
                                                       data = self.moments_per_section.X,
                                                       compression = compression)
                    moments_per_section.create_dataset('Y',
                                                       data = self.moments_per_section.Y,
                                                       compression = compression)
                    moments_per_section.create_dataset('Z',
                                                       data = self.moments_per_section.Z,
                                                       compression = compression)

                    log('Moments per section saved', level = "OK", color = True)

                if hasattr(self, 'drags'):
                    drags = f.create_group('drags')

                    drags.create_dataset('X',
                                         data = self.drags.X,
                                         compression = compression)
                    drags.create_dataset('Y',
                                         data = self.drags.Y,
                                         compression = compression)
                    drags.create_dataset('Z',
                                         data = self.drags.Z,
                                         compression = compression)  
                    log('Viscous drag forces saved', level = "OK", color = True)

                if hasattr(self, 'V_in_cp'):
                    Vcp = f.create_group('velocities_in_cp')
                    Vcp.create_dataset('X',
                                         data = self.V_in_cp.X,
                                         compression = compression)
                    Vcp.create_dataset('Y',
                                         data = self.V_in_cp.Y,
                                         compression = compression)
                    Vcp.create_dataset('Z',
                                       data = self.V_in_cp.Z,
                                         compression = compression)  
                    log('Free stream vectors from collocation points saved', level = "OK", color = True)
                
                if hasattr(self, 'V_in_lift_location' ):
                    Vil = f.create_group('velocities_in_lift_location')
                    Vil.create_dataset('X',
                                         data = self.V_in_lift_location.X,
                                         compression = compression)
                    Vil.create_dataset('Y',
                                         data = self.V_in_lift_location.Y,
                                         compression = compression)
                    Vil.create_dataset('Z',
                                       data = self.V_in_lift_location.Z,
                                         compression = compression)  
                    log('Free stream vectors from lift locations saved', level = "OK", color = True)
                

                log(f"Solution saved to {file_name}", level = "OK", color = True)
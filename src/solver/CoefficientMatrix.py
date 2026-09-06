import numpy as np
from preprocessor.GeometrySets import Mesh2D, VectorSet
from preprocessor import FixedVortexGrid
from solver.biosavart import biosavart_vectorized, VortexCoreModels
from tqdm import tqdm
from utilities import summary


class FreeVortexGrid():
    def __init__(self,
                 start_line:Mesh2D):
        """
        Initiate the part of the vortex grid that can
        move in time

        Parameters
        ----------
        start_line : Mesh2D    
        """
        self.start_line = start_line
        self.vortex_grid = start_line

        
    @staticmethod 
    def mesh_moved_by_vectors(mesh:Mesh2D, 
                              vectors:VectorSet) -> Mesh2D:
        """
        Parameters
        ----------
        mesh : Mesh2D
            Initial geometry
        vectors : VectorSet
            Set of the movements vector

        Returns
        -------
        Mesh2D
            Moved geometry
        """
        return Mesh2D(x_set = mesh.X + vectors.X,
                      y_set = mesh.Y + vectors.Y,
                      z_set = mesh.Z + vectors.Z)


    def update_position(self,
                        induced_velocities:np.ndarray,
                        angular_velocity:float,
                        wind_velocity:float,
                        timestep:float):
        """
        Update positions of the free vortex grid.

        Parameters
        ----------
        induced_velocities : np.ndarray
            Induced velocities on every point in
            the free vortex grid.
        angular_velocity : float
            Angular velocity in the rotor plane [ras/s].
        wind_velocity : float
            Axial velocity to the rotor plane [m/s].
        timestep : float
            Time of the single movement step [s].
        """
        
        old_shape = self.vortex_grid.X.shape
        previous_grid = VectorSet(x_set = self.vortex_grid.X.flatten(),
                                  y_set = self.vortex_grid.Y.flatten(),
                                  z_set = self.vortex_grid.Z.flatten())
        
        # rotational displacement     
        psi = angular_velocity * timestep
        
        rotated_grid = previous_grid.rotated_by_angle(XY = True,
                                                      angle = -psi)
        
        reshaped_grid = Mesh2D(x_set = rotated_grid.X.reshape(old_shape),
                               y_set = rotated_grid.Y.reshape(old_shape),
                               z_set = rotated_grid.Z.reshape(old_shape),)

        # translational displacement
        movement = VectorSet(x_set = induced_velocities.X * timestep,
                             y_set = induced_velocities.Y * timestep,
                             z_set = (induced_velocities.Z + wind_velocity) * timestep)

        new_grid = self.mesh_moved_by_vectors(mesh = reshaped_grid,
                                              vectors = movement)
        
        # adding the fixed starting line
        self.vortex_grid = Mesh2D(x_set = np.vstack([self.start_line.X, new_grid.X]),
                                  y_set = np.vstack([self.start_line.Y, new_grid.Y]),
                                  z_set = np.vstack([self.start_line.Z, new_grid.Z]),)
        

    def update_gamma(self,
                     new_gamma_line:np.ndarray):
        """
        Update the vortex filament strength array 
        by the vortex line values escaping the trailing edge.

        Parameters
        ----------
        new_gamma_line : np.ndarray
            New beginning line of the vortex strengths in
            the free wake. 
        """
        if hasattr(self,'gamma'):
            self.gamma = np.vstack([new_gamma_line, self.gamma])
        else:
            self.gamma = new_gamma_line


    def induced_velocities(self,
                           gamma:np.ndarray,
                           target_points:Mesh2D,
                           wing_symmetry:bool = False,
                           blades:int = 3,
                           core_model:VortexCoreModels = None,
                           verbose:bool = False) -> np.ndarray:
        """
        Compute the velocities in target point 
        induced by the free vortex grid.

        Parameters
        ----------
        gamma : np.ndarray
            Vortex filament strength of the free vortex grid
        target_points : Mesh2D
            Points of the induced velocities evaluation
        wing_symmetry : bool, optional
            Calculate the mirror symmetric vortex grid,
            option for the wing case, 
            by default False
        blades : int, optional
            Number of the considered rotor blades, by default 3
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        verbose : bool, optional
            Print the information about the used core models, by default False
    
        Returns
        -------
        np.ndarray
            Induced velocities in the X direction
        np.ndarray
            Induced velocities in the Y direction
        np.ndarray
            Induced velocities in the Z direction
        """
        
        M = target_points.X.size
        N = self.corners_leading_root.X.size

        U_total = np.zeros((M, N))
        V_total = np.zeros((M, N))
        W_total = np.zeros((M, N))

        edges = [
            (self.corners_leading_root, self.corners_leading_tip),    # RL → TL
            (self.corners_leading_tip, self.corners_trailing_tip),    # TL → TT
            (self.corners_trailing_tip, self.corners_trailing_root),  # TT → RT
            (self.corners_trailing_root, self.corners_leading_root),  # RT → RL
        ]


        edge_labels = ["RL→TL", "TL→TT", "TT→RT", "RT→RL"] # to proper computation tracking at the progress bar tqdm
        
        for j, ((start, end), label) in enumerate(tqdm(zip(edges, edge_labels),
                                                       total = len(edges),
                                                       desc = "filament_sets",
                                                       unit = "filament_set")
                                                       ):
            
            U, V, W = biosavart_vectorized(collocation_points = target_points,
                                           filament_start = start,
                                           filament_end = end,
                                           sym = False,
                                           core_model = core_model,
                                           verbose = verbose)
            U_total += U
            V_total += V
            W_total += W

            if wing_symmetry is True:
                U, V, W = biosavart_vectorized(collocation_points = target_points,
                                                filament_start = start,
                                                filament_end = end,
                                                sym = True,
                                                core_model = core_model,
                                                verbose = verbose)
                U_total -= U
                V_total -= V
                W_total -= W
            
            elif blades > 1:
                
                angle_shift  = 2 * np.pi / blades
                for i in tqdm(range(1, blades),
                              desc =f" other blades [{label}]",
                              leave = False,
                              unit = "blade"):
                    moved_starts  = (VectorSet(x_set = start.X,
                                               y_set = start.Y,
                                               z_set = start.Z).
                                               rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_starts = Mesh2D(x_set=moved_starts.X,
                                        y_set=moved_starts.Y,
                                        z_set=moved_starts.Z)   
                         
                    moved_ends  = (VectorSet(x_set = end.X,
                                             y_set = end.Y,
                                             z_set = end.Z).
                                             rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_ends = Mesh2D(x_set=moved_ends.X,
                                      y_set=moved_ends.Y,
                                      z_set=moved_ends.Z)     

                    U, V, W = biosavart_vectorized(collocation_points = target_points,
                                                        filament_start = new_starts,
                                                        filament_end = new_ends,
                                                        sym = False,
                                                        core_model = core_model,
                                                        verbose = verbose)

                    U_total += U
                    V_total += V
                    W_total += W       

        U_induced = U_total @ gamma 
        V_induced = V_total @ gamma
        W_induced = W_total @ gamma 
        
        return U_induced, V_induced, W_induced


    @property
    def corners_leading_root(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            leading edge and blade root.
        """
        return Mesh2D(x_set = self.vortex_grid.X[:-1,:-1],
                      y_set = self.vortex_grid.Y[:-1,:-1],
                      z_set = self.vortex_grid.Z[:-1,:-1])


    @property
    def corners_leading_tip(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            leading edge and blade tip.
        """
        return Mesh2D(x_set = self.vortex_grid.X[:-1,1:],
                      y_set = self.vortex_grid.Y[:-1,1:],
                      z_set = self.vortex_grid.Z[:-1,1:])


    @property
    def corners_trailing_root(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            trailing edge and blade root.
        """
        return Mesh2D(x_set = self.vortex_grid.X[1:,:-1],
                      y_set = self.vortex_grid.Y[1:,:-1],
                      z_set = self.vortex_grid.Z[1:,:-1])


    @property
    def corners_trailing_tip(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            traiing edge and blade tip.
        """
        return Mesh2D(x_set = self.vortex_grid.X[1:,1:],
                      y_set = self.vortex_grid.Y[1:,1:],
                      z_set = self.vortex_grid.Z[1:,1:])


    @property
    def panel_number(self) -> int:
        """
        Returns
        -------
        int
            Number of the free wake vortex rings.
        """
        return self.corners_leading_root.size


class ComputationPanels():
    def __init__(self, 
                 full_mesh:FixedVortexGrid):
        """
        Initiate the computation panels.

        Parameters
        ----------
        full_mesh : FixedVortexGrid
            Vortex grid representing the lifting surface
            and fixed wake mesh
        """
        self.vortex_grid = full_mesh.xyzqcst
        self.nbs = self.vortex_grid.shape[1] - 1 
        self.collocation_points = full_mesh.collocation_points
        self.collocation_points.X = self.collocation_points.X 
        self.normals = full_mesh.panel_normal_vectors
        

    @staticmethod
    def steady_matrix(entire_matrix:np.ndarray,
                      n_collocation_points:int,
                      target_points:int, 
                      n_spanwise: int) -> np.ndarray:
        """
        Convert the rectangular AIC matrix into square one 
        based on the steady assumption of constant vortex filament strength
        in wake and Kutta condition.

        Parameters
        ----------
        entire_matrix : np.ndarray
            Initial rectangular AIC matrix
        n_collocation_points : int
            Number of the collocation points
        target_points : int
            The required AIC matrix dimension
        n_spanwise : int
            Number of spanwise sections

        Returns
        -------
        np.ndarray
            Square AIC matrix
        """

        wake_matrix = entire_matrix[:, n_collocation_points:]          # (N, n_wake_cols)
        n_wake_rows = wake_matrix.shape[1] // n_spanwise

        wake_matrix = wake_matrix.reshape(target_points, n_wake_rows, n_spanwise)  # (N, n_wake, nbs)
        wake_matrix = wake_matrix.sum(axis=1)                         # (N, nbs) 
        new_matrix = entire_matrix[:, :n_collocation_points].copy()   # (N, N)
        new_matrix[:, -n_spanwise:] += wake_matrix
        return new_matrix

    def compute_influence_matrix(self, 
                                 wing_symmetry:bool = False,
                                 blades:int = 3,
                                 save_geometry_for_plot:bool = False,
                                 different_vortex_source:FreeVortexGrid = None,
                                 core_model:VortexCoreModels = None,
                                 verbose:bool = False) -> np.ndarray:
        """
        Compute aerodynamic influence coefficients
        matrix of the defined influence from
        own or external vortex rings on the 
        collocation points defined in the class.

        Parameters
        ----------
        wing_symmetry : bool, optional
                    Calculate the mirror symmetric vortex grid,
                    option for the wing case, 
                    by default False
        blades : int, optional
            Number of the considered rotor blades, by default 3
        save_geometry_for_plot : bool, optional
            Save rotor and wake geometry in
            the result file, 
            by default False
        different_vortex_source : FreeVortexGrid, optional
            Use different vortex grid definition
            than defined in the class, 
            useful for computing the free wake
            vortex rings influence on the 
            defined collocation points, 
            by default None
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        verbose : bool, optional
            Print the information about the used core models, by default False

        Returns
        -------
        np.ndarray
            AIC matrix
        """
        
        N = self.collocation_points.X.size

        if different_vortex_source is None:
            M = self.corners_leading_root.X.size
        else:
            grid = different_vortex_source
            M = grid.corners_leading_root.X.size
        
        U_total = np.zeros((N, M))
        V_total = np.zeros((N, M))
        W_total = np.zeros((N, M))

        if different_vortex_source is None:
            edges = [
                (self.corners_leading_root, self.corners_leading_tip),    # RL → TL
                (self.corners_leading_tip, self.corners_trailing_tip),    # TL → TT
                (self.corners_trailing_tip, self.corners_trailing_root),  # TT → RT
                (self.corners_trailing_root, self.corners_leading_root),  # RT → RL
            ]
        else:
            edges = [
                (grid.corners_leading_root, grid.corners_leading_tip),    # RL → TL
                (grid.corners_leading_tip, grid.corners_trailing_tip),    # TL → TT
                (grid.corners_trailing_tip, grid.corners_trailing_root),  # TT → RT
                (grid.corners_trailing_root, grid.corners_leading_root),  # RT → RL
            ]
        
        edge_labels = ["RL→TL", "TL→TT", "TT→RT", "RT→RL"] # to proper computation tracking at the progress bar tqdm
        if save_geometry_for_plot:
            if different_vortex_source is None:
                self.entire_mesh = [self.vortex_grid]
            else:
                self.free_wake_mesh = [different_vortex_source.vortex_grid]

        for j, ((start, end), label) in enumerate(tqdm(zip(edges, edge_labels),
                                                       total = len(edges),
                                                       desc = "filament_sets",
                                                       unit = "filament_set")
                                                       ):
            
            U, V, W = biosavart_vectorized(collocation_points = self.collocation_points,
                                                filament_start = start,
                                                filament_end = end,
                                                sym = False,
                                                core_model = core_model,
                                                verbose = verbose)
            U_total += U
            V_total += V
            W_total += W

            if wing_symmetry is True:
                U, V, W = biosavart_vectorized(collocation_points = self.collocation_points,
                                                filament_start = start,
                                                filament_end = end,
                                                sym = True,
                                                core_model = core_model,
                                                verbose = verbose)
                U_total -= U
                V_total -= V
                W_total -= W
            
            elif blades > 1:
                from preprocessor.GeometrySets import VectorSet
                
                angle_shift  = 2 * np.pi / blades
                for i in tqdm(range(1, blades),
                              desc =f" other blades [{label}]",
                              leave = False,
                              unit = "blade"):
                    moved_starts  = (VectorSet(x_set = start.X,
                                               y_set = start.Y,
                                               z_set = start.Z).
                                               rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_starts = Mesh2D(x_set=moved_starts.X,
                                        y_set=moved_starts.Y,
                                        z_set=moved_starts.Z)   
                         
                    moved_ends  = (VectorSet(x_set = end.X,
                                             y_set = end.Y,
                                             z_set = end.Z).
                                             rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_ends = Mesh2D(x_set=moved_ends.X,
                                      y_set=moved_ends.Y,
                                      z_set=moved_ends.Z)     

                    U, V, W = biosavart_vectorized(collocation_points = self.collocation_points,
                                    filament_start = new_starts,
                                    filament_end = new_ends,
                                    sym = False,
                                    core_model = core_model,
                                    verbose = verbose)

                    U_total += U
                    V_total += V
                    W_total += W       

                    if save_geometry_for_plot:
                        if different_vortex_source is None:
                            if j ==0:
                                self.entire_mesh.append(Mesh2D(blank = True,
                                                                size = self.vortex_grid.X.shape))
                                
                                self.entire_mesh[i].X[:-1,:-1] = new_starts.X.reshape(-1,self.nbs) # save the left upper matrix of the coordinates
                                self.entire_mesh[i].X[:-1,-1] = new_ends.X.reshape(-1,self.nbs)[:,-1] # save right upper row of  the matrix
                                
                                self.entire_mesh[i].Y[:-1,:-1] = new_starts.Y.reshape(-1,self.nbs)
                                self.entire_mesh[i].Y[:-1,-1] = new_ends.Y.reshape(-1,self.nbs)[:,-1] # nbs is used because these matrices are cutted byb one so number of nodes -1
                                
                                self.entire_mesh[i].Z[:-1,:-1] = new_starts.Z.reshape(-1,self.nbs)
                                self.entire_mesh[i].Z[:-1,-1] = new_ends.Z.reshape(-1,self.nbs)[:,-1]      
                            elif j==2:

                                self.entire_mesh[i].X[-1,:-1] = new_ends.X.reshape(-1,self.nbs)[-1,:] # save bottom left row of the coordinates
                                self.entire_mesh[i].X[-1,-1] = new_starts.X.reshape(-1,self.nbs)[-1,-1] # save bottom right corner of the coordinates
                                
                                self.entire_mesh[i].Y[-1,:-1] = new_ends.Y.reshape(-1,self.nbs)[-1,:]
                                self.entire_mesh[i].Y[-1,-1] = new_starts.Y.reshape(-1,self.nbs)[-1,-1]
                                
                                self.entire_mesh[i].Z[-1,:-1] = new_ends.Z.reshape(-1,self.nbs)[-1,:]
                                self.entire_mesh[i].Z[-1,-1] = new_starts.Z.reshape(-1,self.nbs)[-1,-1]
                        else:
                            if j ==0:
                                self.free_wake_mesh.append(Mesh2D(blank = True,
                                                                size = different_vortex_source.vortex_grid.X.shape))

                                self.free_wake_mesh[i].X[:-1,:-1] = new_starts.X.reshape(-1,self.nbs) # save the left upper matrix of the coordinates
                                self.free_wake_mesh[i].X[:-1,-1] = new_ends.X.reshape(-1,self.nbs)[:,-1] # save right upper row of  the matrix
                                
                                self.free_wake_mesh[i].Y[:-1,:-1] = new_starts.Y.reshape(-1,self.nbs)
                                self.free_wake_mesh[i].Y[:-1,-1] = new_ends.Y.reshape(-1,self.nbs)[:,-1] # nbs is used because these matrices are cutted byb one so number of nodes -1
                                
                                self.free_wake_mesh[i].Z[:-1,:-1] = new_starts.Z.reshape(-1,self.nbs)
                                self.free_wake_mesh[i].Z[:-1,-1] = new_ends.Z.reshape(-1,self.nbs)[:,-1]      
                            elif j==2:

                                self.free_wake_mesh[i].X[-1,:-1] = new_ends.X.reshape(-1,self.nbs)[-1,:] # save bottom left row of the coordinates
                                self.free_wake_mesh[i].X[-1,-1] = new_starts.X.reshape(-1,self.nbs)[-1,-1] # save bottom right corner of the coordinates
                                
                                self.free_wake_mesh[i].Y[-1,:-1] = new_ends.Y.reshape(-1,self.nbs)[-1,:]
                                self.free_wake_mesh[i].Y[-1,-1] = new_starts.Y.reshape(-1,self.nbs)[-1,-1]
                                
                                self.free_wake_mesh[i].Z[-1,:-1] = new_ends.Z.reshape(-1,self.nbs)[-1,:]
                                self.free_wake_mesh[i].Z[-1,-1] = new_starts.Z.reshape(-1,self.nbs)[-1,-1]
                        

        nx = self.normals.X.ravel()[:, np.newaxis] # (N, 1)
        ny = self.normals.Y.ravel()[:, np.newaxis]
        nz = self.normals.Z.ravel()[:, np.newaxis]

        if different_vortex_source is None: # frozen wake case with assumption wake panel with the same gamma as wing panels
            U_steady = self.steady_matrix(entire_matrix = U_total, n_collocation_points = N, target_points = N, n_spanwise = self.nbs)
            V_steady = self.steady_matrix(entire_matrix = V_total, n_collocation_points = N, target_points = N, n_spanwise = self.nbs)
            W_steady = self.steady_matrix(entire_matrix = W_total, n_collocation_points = N, target_points = N, n_spanwise = self.nbs)

            self.U_steady = U_steady
            self.V_steady = V_steady
            self.W_steady = W_steady
            
            A = U_steady * nx + V_steady * ny + W_steady * nz

        else: 
            A = U_total * nx + V_total * ny + W_total * nz


        summary("AIC matrix data", {
                "AIC matrix size": A.shape,
                "Number of collocation points": N,
                "Number of all filaments":  3 * 4 * (self.vortex_grid.shape[1] - 1 ) * (self.vortex_grid.shape[0] - 1) 
        })

        return A


    @property
    def corners_leading_root(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            leading edge and blade root.
        """
        return Mesh2D(x_set = self.vortex_grid.X[:-1,:-1],
                      y_set = self.vortex_grid.Y[:-1,:-1],
                      z_set = self.vortex_grid.Z[:-1,:-1])


    @property
    def corners_leading_tip(self):
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            leading edge and blade tip.
        """
        return Mesh2D(x_set = self.vortex_grid.X[:-1,1:],
                      y_set = self.vortex_grid.Y[:-1,1:],
                      z_set = self.vortex_grid.Z[:-1,1:])


    @property
    def corners_trailing_root(self):
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            trailing edge and blade root.
        """
        return Mesh2D(x_set = self.vortex_grid.X[1:,:-1],
                      y_set = self.vortex_grid.Y[1:,:-1],
                      z_set = self.vortex_grid.Z[1:,:-1])


    @property
    def corners_trailing_tip(self):
        """
        Returns
        -------
        Mesh2D
            Vortex ring corners from the side of the
            trailing edge and blade tip.
        """
        return Mesh2D(x_set = self.vortex_grid.X[1:,1:],
                      y_set = self.vortex_grid.Y[1:,1:],
                      z_set = self.vortex_grid.Z[1:,1:])


    @property
    def panel_number(self):
        """
        Returns
        -------
        int
            Number of the free wake vortex rings.
        """
        return self.corners_leading_root.size


    def induced_velocities(self,
                           gamma:np.ndarray,
                           target_points:Mesh2D,
                           wing_symmetry:bool = False,
                           blades:int = 3,
                           core_model:VortexCoreModels = None,
                           verbose:bool = False):
        """
        Compute the velocities in target point 
        induced by the fixed vortex grid.

        Parameters
        ----------
        gamma : np.ndarray
            Vortex filament strength of the fixed vortex grid
        target_points : Mesh2D
            Points of the induced velocities evaluation
        wing_symmetry : bool, optional
            Calculate the mirror symmetric vortex grid,
            option for the wing case, 
            by default False
        blades : int, optional
            Number of the considered rotor blades, by default 3
        core_model : VortexCoreModels, optional
            Vortex core model used for the fixed vortex rings, by default None
        verbose : bool, optional
            Print the information about the used core models, by default False
    
        Returns
        -------
        np.ndarray
            Induced velocities in the X direction
        np.ndarray
            Induced velocities in the Y direction
        np.ndarray
            Induced velocities in the Z direction
        """
        
        N = target_points.X.size
        M = self.corners_leading_root.X.size

        U_total = np.zeros((N, M))
        V_total = np.zeros((N, M))
        W_total = np.zeros((N, M))

        edges = [
            (self.corners_leading_root, self.corners_leading_tip),    # RL → TL
            (self.corners_leading_tip, self.corners_trailing_tip),    # TL → TT
            (self.corners_trailing_tip, self.corners_trailing_root),  # TT → RT
            (self.corners_trailing_root, self.corners_leading_root),  # RT → RL
        ]


        edge_labels = ["RL→TL", "TL→TT", "TT→RT", "RT→RL"] # to proper computation tracking at the progress bar tqdm
        
        for j, ((start, end), label) in enumerate(tqdm(zip(edges, edge_labels),
                                                       total = len(edges),
                                                       desc = "filament_sets",
                                                       unit = "filament_set")
                                                       ):
            
            U, V, W = biosavart_vectorized(collocation_points = target_points,
                                                filament_start = start,
                                                filament_end = end,
                                                sym = False,
                                                core_model = core_model,
                                                verbose = verbose)
            U_total += U
            V_total += V
            W_total += W

            if wing_symmetry is True:
                U, V, W = biosavart_vectorized(collocation_points = target_points,
                                                filament_start = start,
                                                filament_end = end,
                                                sym = True,
                                                core_model = core_model,
                                                verbose = verbose)
                U_total -= U
                V_total -= V
                W_total -= W
            
            elif blades > 1:
                from preprocessor.GeometrySets import VectorSet
                
                angle_shift  = 2 * np.pi / blades
                for i in tqdm(range(1, blades),
                              desc =f" other blades [{label}]",
                              leave = False,
                              unit = "blade"):
                    
                    moved_starts  = (VectorSet(x_set = start.X,
                                               y_set = start.Y,
                                               z_set = start.Z).
                                               rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_starts = Mesh2D(x_set=moved_starts.X,
                                        y_set=moved_starts.Y,
                                        z_set=moved_starts.Z)   
                         
                    moved_ends  = (VectorSet(x_set = end.X,
                                             y_set = end.Y,
                                             z_set = end.Z).
                                             rotated_by_angle(XY=True, angle=i * angle_shift, mesh=True))
                    new_ends = Mesh2D(x_set=moved_ends.X,
                                      y_set=moved_ends.Y,
                                      z_set=moved_ends.Z)     

                    U, V, W = biosavart_vectorized(collocation_points = target_points,
                                                        filament_start = new_starts,
                                                        filament_end = new_ends,
                                                        sym = False,
                                                        core_model = core_model,
                                                        verbose = verbose)

                    U_total += U
                    V_total += V
                    W_total += W       

        n_collocation_points = self.collocation_points.size

        U_steady = self.steady_matrix(entire_matrix = U_total, n_collocation_points = n_collocation_points, target_points = N, n_spanwise = self.nbs)
        V_steady = self.steady_matrix(entire_matrix = V_total, n_collocation_points = n_collocation_points, target_points = N, n_spanwise = self.nbs)
        W_steady = self.steady_matrix(entire_matrix = W_total, n_collocation_points = n_collocation_points, target_points = N, n_spanwise = self.nbs)

        U_induced = U_steady @ gamma 
        V_induced = V_steady @ gamma
        W_induced = W_steady @ gamma 
        
        return U_induced, V_induced, W_induced
import numpy as np
import pandas as pd

from preprocessor.GeometrySets import Vector, Beam, CamberLine, VectorSet, Mesh2D, PanelSet
from preprocessor.GeometrySets import rotation_matrix_by_angle
from preprocessor.InBuild_expon import rotation_matrix_rodrigues
from preprocessor.compute_camber_line import CamberLineInterpolator
from utilities import summary


class StructuralGeometry:
    """
    Structural geometry of the lifting surface

    Loads and stores the geometric properties
    """
    def __init__(self, 
                 xlsx_path:str, 
                 cut_off_thickness:float = 100):
        """
        Initiate the structural geometry

        Parameters
        ----------
        xlsx_path : str
            Path to the xlsx input path
        cut_off_thickness : float, optional
            Relative thickness from which the lifting surface modelling is started [%],
            by default 100
        """
        
        # import the xlsx input file path
        data = pd.read_excel(xlsx_path)

        # import relative thickness and the defined irfoil names
        relative_thickness = np.array(data['Relative thickness [-]'])
        airfoil_types = data['Airfoil type'].dropna()
        
        # create an interpolator and interpolate the camber surface for the vortex sheet
        interpolator = CamberLineInterpolator(airfoil_types = airfoil_types)
        camber_x, camber_z = interpolator.compute_camber_lines(relative_thickness,
                                                               cut_off_thickness = cut_off_thickness)

        # create the camber surface
        valid_section_number = camber_x.shape[1]
        self.relative_thickness = relative_thickness[-valid_section_number:]
        self.camber = CamberLine(x_set = camber_x,
                                 y_set = np.zeros_like(camber_x),
                                 z_set = camber_z)

        # create beam
        # different coordinates with respect to input
        # x -> x - chordwise
        # z -> y - axial
        # y -> z - spanwise
        self.beam_global = Beam(x_set = np.array(data['x [m]'])[-valid_section_number:],
                                y_set = np.array(data['z [m]'])[-valid_section_number:],
                                z_set = np.array(data['y [m]'])[-valid_section_number:])

        # transform the beam to its local coordinate system
        self.beam_local = self.beam_global.in_local_CS()
        
        # import the chord for considerad sections
        self.chord = np.array(data['Chord [m]'])[-valid_section_number:]

        # import the chordwise distance from the trailing eddge to the beam for considered sections
        self.axis_ref_X = np.array(data['Pitch axis aft LE (x/c) [-]'])[-valid_section_number:]  * self.chord
        
        # import the twist angle for considered sections
        self.twist_initial = np.deg2rad(np.array(data['Twist [deg]'])[-valid_section_number:])

        # setup global and local transformation matrices
        self.R_g2l, self.R_l2g = self.beam_global.setup_global_local_transform()

        summary("Structural geometry data", {
            "Chord min [m]": np.min(self.chord),
            "Chord max [m]": np.max(self.chord),
            "Twist min [deg]": np.rad2deg(np.min(self.twist_initial)),
            "Twist max [deg]": np.rad2deg(np.max(self.twist_initial))
        })


class AerodynamicGeometry:
    """
    Aerodynamic geometry of the lifting surface
    
    Loads the structural geometry and creates its aerodynamic discretization
    """
    def __init__(self,
                 structural_geometry:StructuralGeometry,
                 mesh_density:tuple,
                 aoa:float = 0,
                 cosine_spacing:tuple = (False, False)):
        """
        Initiate the aerodynamic geometry

        Parameters
        ----------
        structural_geometry : StructuralGeometry
            Structural geometry input
        mesh_density : tuple(int,int)
            Number of mesh elements (chordwise, spanwise)
        cosine_spacing : tuple(bool,bool), optional
            Applying the cosine spacing (chordwise, spanwise), by default (False, False)
        """
        
        # save the cosine spacing setting
        self.cosine_spacing = cosine_spacing

        # define the chordwise mesh spacing
        if cosine_spacing[0] == True:
            i = np.linspace(0, np.pi, mesh_density[0] + 1)
            self.positions_chordwise = (1 - np.cos(i)) / 2
        else: 
            self.positions_chordwise = np.linspace(0, 1, mesh_density[0] + 1)
        
        # define the spanwise mesh spacing
        if cosine_spacing[1] == True:
            j = np.linspace(0, np.pi, mesh_density[1] + 1)
            self.positions_spanwise = (1 - np.cos(j)) / 2
        else:
            self.positions_spanwise = np.linspace(0, 1, mesh_density[1] + 1)

        # save the number of elements    
        self.n_spanwise = len(self.positions_spanwise)
        self.n_chordwise = len(self.positions_chordwise)

        # save the angle of attack
        self.aoa = aoa
        
        #save the structural input 
        self.structural = structural_geometry

        # save the rotation matrices
        self.R_g2l, self.R_l2g = self.structural.R_g2l, self.structural.R_l2g
        
        # crate the elastic axis with an aerodynamic discretization
        self.beam_local = Beam(blank = True,
                               size = self.n_spanwise)
        self.beam_local.Y = self.positions_spanwise * self.structural.beam_local.Y[-1]
        self.beam_local.X, self.beam_local.Z = self.interpolationXZ(self.beam_local,
                                                                    self.structural.beam_local,
                                                                    self.structural.beam_local)
        # transform the elastic axis to the global coordinates
        self.beam_global = self.beam_local.moved_to_('global', 
                                                     reference_node = self.structural.beam_global.get_node(index = 0),
                                                     rotation_matrix = self.R_l2g)

        # create the aerodynamic chord distribution
        self.chord = np.interp(self.beam_local.Y, self.structural.beam_local.Y, self.structural.chord)

        summary("Aerodynamic geometry data", {
            "Number of mesh elements spanwise": self.n_spanwise - 1,
            "Number of mesh elements chordwise": self.n_chordwise - 1,
            "Pitch angle [deg]": self.aoa,
        })       


    @property
    def between_section_chords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Average middle chord of each chordwise element line
        """
        return (self.chord[1:] + self.chord[:-1]) / 2 
    
    
    @property
    def between_section_areas(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Area of each chordwise element line
        """
        return self.between_section_chords * np.diff(self.beam_local.Y)


    @property
    def mean_aerodynamic_chords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Mean aerodynamic chord of each chordwise element line
        """
        ratio = self.chord[1:] / self.chord[:-1]
        return (2/3) * self.chord[:-1] * (ratio ** 2 + ratio + 1) / (ratio + 1)


    @property
    def mean_aerodynamic_chord_locations(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Spanwise locations of mean aerodynamic chords for each chordwise element line
        """
        loc_mac = np.zeros(self.n_spanwise - 1)
        for i in range(len(loc_mac)):
            if np.isclose(self.chord[i], self.chord[i+1]):
                loc_mac[i] = (self.beam_local.Y[i] + self.beam_local.Y[i+1]) * 0.5
            else:
                loc_mac[i] = np.interp(
                self.mean_aerodynamic_chords[i],
                [self.chord[i+1], self.chord[i]], 
                [self.beam_local.Y[i+1], self.beam_local.Y[i]])
        return loc_mac
        

    @staticmethod
    def interpolationXZ(beam_aero_local:Beam, 
                        beam_struct_local:Beam, 
                        interpolation_input:np.ndarray) -> np.ndarray:
        """
        Interpolate the input from structural
        to aerodynamic beam, based on the Y (spanwise)
        positions, on X (chordwise) and Z 
        (perpendicular to chordwise) direction

        Parameters
        ----------
        beam_aero_local : Beam
            Aerodynamic beam - reference interpolation target
        beam_struct_local : Beam
            Structural beam - reference interpolation source
        interpolation_input : np.ndarray
            Input to interpolate

        Returns
        -------
        np.ndarray
            Interpolated property on X direction
        np.ndarray
            Interpolated property on Z direction
        """
        X = np.interp(beam_aero_local.Y,beam_struct_local.Y,interpolation_input.X)
        Z = np.interp(beam_aero_local.Y,beam_struct_local.Y,interpolation_input.Z)
        return X,Z


    @staticmethod
    def interpolation(beam_aero_local:Beam, 
                      beam_struct_local:Beam, 
                      interpolation_input:np.ndarray) -> np.ndarray:
        """
        Interpolate the input from structural
        to aerodynamic beam, based on the Y (spanwise)
        positions, on X (chordwise), Y (spanwise) and Z 
        (perpendicular to chordwise) direction

        Parameters
        ----------
        beam_aero_local : Beam
            Aerodynamic beam - reference interpolation target
        beam_struct_local : Beam
            Structural beam - reference interpolation source
        interpolation_input : np.ndarray
            Input to interpolate

        Returns
        -------
        
        Returns
        -------
        np.ndarray
            Interpolated property on X direction
        np.ndarray
            Interpolated property on Y direction
        np.ndarray
            Interpolated property on Z direction
        """
        X = np.interp(beam_aero_local.Y,beam_struct_local.Y,interpolation_input.X)
        Y = np.interp(beam_aero_local.Y,beam_struct_local.Y,interpolation_input.Y)
        Z = np.interp(beam_aero_local.Y,beam_struct_local.Y,interpolation_input.Z)
        return X,Y,Z


    @property
    def mesh_undeflected(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Geometrical representation of undeflected lifting surface
        """
        
        # interpolate the reference distance for the aerodynamic mesh
        self.axis_ref_X = np.interp(self.beam_local.Y, 
                                    self.structural.beam_local.Y, 
                                    self.structural.axis_ref_X)

        # interpolate the camber surface
        from scipy.interpolate import interp1d
        self.camber = CamberLine(blank = True,
                                 size = (self.structural.camber.shape[0], 
                                         self.n_spanwise))

        interp_x = interp1d(self.structural.beam_local.Y, # (M,)
                            self.structural.camber.X,     # (M, N)
                            axis=1,                       # Interpolate along rows
                            kind='linear',
                            fill_value='extrapolate')
        self.camber.X = self.chord * interp_x(self.beam_local.Y)  # (O, N)

        interp_z = interp1d(self.structural.beam_local.Y,
                            self.structural.camber.Z,
                            axis=1,
                            fill_value='extrapolate')
        self.camber.Z = self.chord * interp_z(self.beam_local.Y)

        # construct the camberline surface
        real_positions_chordwise = np.outer(self.positions_chordwise, 
                                            self.chord)
        
        camz = np.zeros_like(real_positions_chordwise)

        for section in range(self.n_spanwise):
            camz[:, section] = np.interp(real_positions_chordwise[:, section],                          # x – nowe punkty, gdzie chcemy wartość
                                         self.camber.X[:, section] - self.camber.X[0,section],          # xp – znane punkty na osi x
                                         self.camber.Z[:, section])                                     # fp – wartości w tych punktach
             
        # create the undeflected mesh
        mesh = Mesh2D(blank = True,
                      size = (self.n_chordwise, 
                              self.n_spanwise))

        for section, reference_axis in enumerate(self.axis_ref_X):
            for chordwise_index, pos_chord in enumerate(real_positions_chordwise[:,section]):
                
                caml = np.array([0,0,camz[chordwise_index,section]])
                camg = self.R_l2g @ caml

                mesh.X[chordwise_index,section] = self.beam_global.X[section] + camg[0] + pos_chord - reference_axis
                mesh.Y[chordwise_index,section] = self.beam_global.Y[section] + camg[1] 
                mesh.Z[chordwise_index,section] = self.beam_global.Z[section] + camg[2] 
        
        return mesh


    @property
    def mesh_deflected(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Geometrical representation of deflected lifting surface,
            including twist design angle and the twist angle coming 
            from the structural deflections.
        """
        
        # twist distribution
        self.twist_initial = np.interp(self.beam_local.Y, 
                                       self.structural.beam_local.Y, 
                                       self.structural.twist_initial)
        
        self.chord_bef_def = Vector([1,0,0])
        chords_bef_def_set = VectorSet(one_vector_input = True,
                                       one_vector = np.tile(self.chord_bef_def, 
                                                            self.n_spanwise))

        self.normal_vectors_global = (self.beam_local.
                                      normal_vectors(plane = "YZ").
                                      rotated_by_matrix(self.R_l2g))

        from scipy.sparse import block_diag

        matrices_twist_initial = np.stack([rotation_matrix_by_angle(a) for a in self.twist_initial])

        RotMatrix_twist_initial = block_diag(matrices_twist_initial, 
                                             format = 'csr')

        self.chords_aft_def_set =  (chords_bef_def_set.
                                    rotated_by_matrix(self.R_g2l).
                                    rotated_by_matrix(RotMatrix_twist_initial, big_matrix = True).
                                    rotated_by_matrix(self.R_l2g))

        local_spanwise_vecs = self.normal_vectors_global.cross_with_single(vec = self.chord_bef_def.
                                                                                 rotated_by_angle(-self.aoa))

        self.local_spanwise_vecs = local_spanwise_vecs


        self.local_chordwise_vecs = self.chords_aft_def_set.projected_on_planes(plane_normals = local_spanwise_vecs).normalized


        mesh_deflected = Mesh2D(blank = True,
                                size = self.mesh_undeflected.shape)

        for section, normal_vector in enumerate(self.normal_vectors_global.as_matrix):
            for chordwise_index in range(self.n_chordwise):
                chordwise_shift = self.local_chordwise_vecs.get_vector(section) * (self.mesh_undeflected.X[chordwise_index, section] - self.beam_global.X[section])
                normal_shift = normal_vector * (self.mesh_undeflected.Z[chordwise_index, section] - self.beam_global.Z[section])
                final_shift = chordwise_shift + normal_shift

                mesh_deflected.X[chordwise_index, section] = self.beam_global.X[section] + final_shift[0]
                mesh_deflected.Y[chordwise_index, section] = self.beam_global.Y[section] + final_shift[1]
                mesh_deflected.Z[chordwise_index, section] = self.beam_global.Z[section] + final_shift[2]
        
        return mesh_deflected


    @property
    def mesh_deflected_rotated(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Geometrical representation of deflected rotated by angle of attack
        """
        previous_root = (Vector(vector = self.structural.beam_global.get_node(index = 0)))
        new_root = previous_root.rotated_by_angle(self.aoa)

        mesh_deflected = self.mesh_deflected

        mesh_size = mesh_deflected.size

        delta_vectors = mesh_deflected.as_vector - np.tile(previous_root, mesh_size)
        moved_coords = (VectorSet(one_vector_input = True,
                                  one_vector = delta_vectors).
                                  rotated_by_angle(self.aoa).
                                  as_vector
                                  + np.tile(new_root, mesh_size))
        
        mesh_deflected_rotated = Mesh2D(one_vector_input = True,
                                        one_vector = moved_coords)
        mesh_deflected_rotated.reshape_set(new_shape = [self.n_chordwise, self.n_spanwise])
        
        return mesh_deflected_rotated
        
   
    @property
    def relative_thickness_of_mean_aerodynamic_chords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Relative thickness of the mean aerodynamic chords
        """
        return np.interp(self.mean_aerodynamic_chord_locations, self.structural.beam_local.Y, self.structural.relative_thickness)


    @property
    def relative_thickness_of_between_section_chords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Relative thickness of the average between section chord
        """
        between_section_Y = (self.beam_local.Y[:-1] + self.beam_local.Y[1:]) / 2
        return np.interp(between_section_Y, self.structural.beam_local.Y, self.structural.relative_thickness)


class FixedVortexGrid:
    """Vortex rings representation which is fixed in space
    for the entire analysis

    Loads the Aerodynamic Geometry and creates the vortex grid representation,
    with all of the blade elements and the first element row of the wake.
    """
    def __init__(self, 
                 aerodynamic_geometry:AerodynamicGeometry,
                 origin:np.ndarray,
                 wind_velocity:float, 
                 angular_velocity:float,
                 timestep:float):
        """
        Initialization of the vortex grid which is fixed 
        in space through the entire analysis.

        Parameters
        ----------
        aerodynamic_geometry : AerodynamicGeometry
            Aerodynamic discretization of the lifting surface geometry
        origin : np.ndarray
            Origin of the global coordinate system
        wind_velocity : float
            Axial velocity used to fixed mesh wake movement
        angular_velocity : float
            Angular velocity used to fixed mesh wake movement
        timestep : float
            Timestep used to fixed mesh wake movement
        """

        # save for possible using in the vortex diffusion implementation
        self.blade_geometry = aerodynamic_geometry
        self.timestep = timestep

        # aerodynamic mesh located in the global coordinate system
        wing_mesh = self.mesh_moved_by_vectors(mesh = aerodynamic_geometry.mesh_deflected_rotated, 
                                               vectors = VectorSet(one_vector_input = True,
                                                                   one_vector = - origin))

        # vector set representing the streamwise direction from the trailing edge
        vectors = VectorSet(x_set = wing_mesh.X[-1] - wing_mesh.three_quarter_chords.X[-1],
                            y_set = wing_mesh.Y[-1] - wing_mesh.three_quarter_chords.Y[-1],
                            z_set = wing_mesh.Z[-1] - wing_mesh.three_quarter_chords.Z[-1])

        first_wake_line = VectorSet(x_set = wing_mesh.X[-1] + vectors.X,
                                    y_set = wing_mesh.Y[-1] + vectors.Y,
                                    z_set = wing_mesh.Z[-1] + vectors.Z) 

        # spatial step of the wake        
        psi = angular_velocity * timestep
        delta_z = wind_velocity * timestep
        
        new_line = first_wake_line.rotated_by_angle(XY = True, angle = - psi)
        new_line.Z = new_line.Z + delta_z

        fixed_wake_mesh = Mesh2D(x_set = np.vstack([first_wake_line.X, new_line.X]),
                                 y_set = np.vstack([first_wake_line.Y, new_line.Y]),
                                 z_set = np.vstack([first_wake_line.Z, new_line.Z]))

        self.wing_mesh = wing_mesh
        self.fixed_wake_mesh = fixed_wake_mesh

        self.trailin_edge = Mesh2D(x_set = self.wing_mesh.X[-1,:],
                                   y_set = self.wing_mesh.Y[-1,:],
                                   z_set = self.wing_mesh.Z[-1,:])
                
        self.panels = PanelSet(self.wing_mesh)
        self.panel_widths = aerodynamic_geometry.beam_local._distance_between_sections

        self.cosine_spacing = aerodynamic_geometry.cosine_spacing


    @property
    def collocation_points(self) -> Mesh2D:
        """    
        Returns
        -------
        Mesh2D
            Places of imposing the non-penetration boundary conditions,
            based on the lumped vortex element assumption.

        Reference
        -------
            Katz n' Plotkin "Low-speed Aerodynamics - second edition"
            Chapter 5.5
        """
        return self.wing_mesh.three_quarter_chords.spanwise_midpoints
    
    
    @property
    def steady_force_locations(self) -> Mesh2D:
        """    
        Returns
        -------
        Mesh2D
            Locations of the lift forces.
            Assumed at the spanwise filament centres.
        """
        return self.wing_mesh.quarter_chords.spanwise_midpoints
    

    @property  
    def gamma_vectors(self) -> VectorSet:
        """    
        Returns
        -------
        VectorSet
            Spanwise filament vectors, required for the
            Kutta - Joukowski calculation
        """
        return self.wing_mesh.quarter_chords.spanwise_vector_set


    @property
    def panel_aeras(self) -> np.ndarray:
        """
        Calculates the panel areas based on the diagonals cross products

        Returns
        -------
        np.ndarray
            Panel areas
        """    
        return self.panels.areas


    @property
    def panel_normal_vectors(self) -> VectorSet:
        """
        Returns
        -------
        VectorSet
            Panel normals
        """
        return self.panels.normals


    @property
    def xyzqcst(self) -> Mesh2D:
        """
        Returns
        -------
        Mesh2D
            Vortex grid point representation of both lifting surface and wake
        """
        return Mesh2D(x_set = np.vstack([self.wing_mesh.quarter_chords.X, self.fixed_wake_mesh.X]),
                      y_set = np.vstack([self.wing_mesh.quarter_chords.Y, self.fixed_wake_mesh.Y]),
                      z_set = np.vstack([self.wing_mesh.quarter_chords.Z, self.fixed_wake_mesh.Z]))


    @staticmethod
    def mesh_moved_by_vectors(mesh:Mesh2D, 
                              vectors:VectorSet) -> Mesh2D:
        """
        Moves the mesh by set of vectors

        Parameters
        ----------
        mesh : Mesh2D
            Initial mesh set 
        vectors : VectorSet
            Vector set that moves all of the mesh points 
            based on the index

        Returns
        -------
        Mesh2D
            Moved mesh set
        """
        return Mesh2D(x_set = mesh.X + vectors.X,
                      y_set = mesh.Y + vectors.Y,
                      z_set = mesh.Z + vectors.Z)

    
    @property 
    def vortex_lifetime(self) -> np.ndarray:
        """
        Lifetime required for the vortex diffusion implementation.

        Returns
        -------
        np.ndarray
            Lifetime of the vortex rings at blade and in the wake
        """

        #array of the lifting surface vortex age in time
        blade_time = np.zeros((self.blade_geometry.n_chordwise - 1, self.blade_geometry.n_spanwise - 1))

        # averarged time age of the fixed vortex element ring, (t + 0) / 2 
        fixed_panel_time = self.timestep / 2 
        wake_time = np.ones((1,self.blade_geometry.n_spanwise - 1)) * fixed_panel_time

        return np.vstack([blade_time, wake_time])


    @property   
    def vortex_local_chords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Chord lengths related to the vortex rings
        """
        local_c_spanwise = self.blade_geometry.between_section_chords 
        dimension = self.xyzqcst.shape[0] - 1                         

         # (dimension, n_spanwise - 1)
        return np.tile(local_c_spanwise, (dimension, 1))              
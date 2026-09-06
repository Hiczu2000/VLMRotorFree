import pyvista as pv
import numpy as np
import h5py
from utilities import log, summary, section
from preprocessor import Mesh2D, VectorSet

class PostProcessor:
    """
    Imports data from .h5 results file
    and stores it in class is also able to 
    visualize and print the results            
    """
    def __init__(self,
                 h5file:str):
        """
        Initialize the postprocesor

        Parameters
        ----------
        h5file : str
            .h5 file path
        """

        section("Postprocessor")
        log(f"Solution loading...", level = "STEP")

        with h5py.File(h5file,'r') as f:
            blades_number = f.attrs['number_of_blades']
            
            entire_mesh = []
            for i in range(blades_number):
                mesh = Mesh2D(x_set = f[f'geometry/blade_{i+1}/X'][:],
                              y_set = f[f'geometry/blade_{i+1}/Y'][:],
                              z_set = f[f'geometry/blade_{i+1}/Z'][:])
                entire_mesh.append(mesh)
            
            self.entire_mesh = entire_mesh
            
            if 'geometry/free_wake_1' in f:
                free_wake = []
                for i in range(blades_number):
                    mesh = Mesh2D(x_set = f[f'geometry/free_wake_{i+1}/X'][:],
                                y_set = f[f'geometry/free_wake_{i+1}/Y'][:],
                                z_set = f[f'geometry/free_wake_{i+1}/Z'][:])
                    free_wake.append(mesh)
            
                self.free_wake = free_wake

            self.gamma_results = f['gamma/gamma'][:]

            # free wake gamma
            if 'fw_gamma' in f:
                self.fw_gamma = f['fw_gamma/fw_gamma'][:]

            self.lifts_results = VectorSet(x_set = f['lifts/X'][:],
                                            y_set = f['lifts/Y'][:],
                                            z_set = f['lifts/Z'][:])
            
            if 'drags' in f:
                self.drag_results = VectorSet(x_set = f['drags/X'][:],
                                                y_set = f['drags/Y'][:],
                                                z_set = f['drags/Z'][:])

            if 'air_density' in f.attrs:
                self.air_density = f.attrs['air_density']
            
            self.V = f.attrs['wind_velocity']
            if 'omega' in f.attrs:
                self.omega = f.attrs['omega'] 

            self.blades = f.attrs['number_of_blades']
            self.blade_chordwise_size = f.attrs['blade_chordwise_mesh_elements'] 
            self.blade_spanwise_size = f.attrs['blade_spanwise_mesh_elements'] 
            
            if 'wake_number_of_rotations' in f.attrs:
                self.rotation_number = f.attrs['wake_number_of_rotations']  
            if 'wake_mesh_elements_per_rotation' in f.attrs:
                self.wake_mesh_elements_per_rotatio =  f.attrs['wake_mesh_elements_per_rotation'] 
            if 'wake_angular_velocity' in f.attrs:
                self.wake_angular_velocity = f.attrs['wake_angular_velocity'] 
            if 'wake_axial_velocity' in f.attrs:
                self.wake_axial_velocity = f.attrs['wake_axial_velocity'] 
            
            self.panel_widths = f.attrs['panel_widths']
            
            if "cosine_spacing" in f.attrs:
                self.cosine_spacing = f.attrs["cosine_spacing"]
            else:
                log(f" Cosine spacing definition was not found in the {h5file}", "WARN", color = True)
            
            if "geometry/panel_normals" in f:
                self.panel_normals = VectorSet(x_set = f['geometry/panel_normals/X'][:],
                                                y_set = f['geometry/panel_normals/Y'][:],
                                                z_set = f['geometry/panel_normals/Z'][:])
            else:
                log(f" Panel normals were not found in the {h5file}", "WARN", color = True)
            
            if "geometry/trailing_edge" in f:
                self.trailing_edge = VectorSet(x_set = f['geometry/trailing_edge/X'][:],
                                                y_set = f['geometry/trailing_edge/Y'][:],
                                                z_set = f['geometry/trailing_edge/Z'][:])
            else:
                log(f" Trailing edge geometry was not found in the {h5file}", "WARN", color = True)
            
            if "velocities_in_cp" in f:
                self.Vcp = VectorSet(x_set = f['velocities_in_cp/X'][:],
                                        y_set = f['velocities_in_cp/Y'][:],
                                        z_set = f['velocities_in_cp/Z'][:])
            else:
                log(f" Free stream velocites at the collocation points were not found in the {h5file}", "WARN", color = True)
            
            if "velocities_in_lift_location" in f:
                self.Vil = VectorSet(x_set = f['velocities_in_lift_location/X'][:],
                                        y_set = f['velocities_in_lift_location/Y'][:],
                                        z_set = f['velocities_in_lift_location/Z'][:])
            else:
                log(f" Free stream velocites at the lift force locations were not found in the {h5file}", "WARN", color = True)
            
            
            if "geometry/gamma_vec" in f:
                self.gamma_vec = VectorSet(x_set = f['geometry/gamma_vec/X'][:],
                                            y_set = f['geometry/gamma_vec/Y'][:],
                                            z_set = f['geometry/gamma_vec/Z'][:])                
            else:
                log(f" Circulation vectors were not found in the {h5file}", "WARN", color = True)
            
        if hasattr(self, 'wake_axial_velocity'):
            summary("Result parameters", {
            "Wind velocity [m/s]": self.V,
            "Wake axial velocity [m/s]": self.wake_axial_velocity,
            "Axial induction factor [-]": (self.V - self.wake_axial_velocity) / self.V,
            "Wake angular velocity [m/s]": self.wake_angular_velocity,
            "Blade number":  self.blades,
            "Blade chordwise mesh elements": self.blade_chordwise_size -1,
            "Blade spanwise mesh elements": self.blade_spanwise_size -1,
            "Wake mesh elements per rotation": self.wake_mesh_elements_per_rotatio,
            "Wake number of rotations": self.rotation_number})

        log(f"Solution loaded from {h5file}", level = "OK", color = True)

        self.blade_meshes, self.wake_meshes = [], []

        for mesh in self.entire_mesh:
            blade_mesh, wake_mesh = mesh.split_set_by_row(row_number = self.blade_chordwise_size)
            self.blade_meshes.append(blade_mesh)
            self.wake_meshes.append(wake_mesh)  


    def free_wake_plot(self):
        """
        Shows the entire geometry 
        of all vortex rings.
        """
        if hasattr(self, 'free_wake'):
            meshes_free = [self.free_wake[0], self.free_wake[1], self.free_wake[2]]
            meshes_entire = [self.entire_mesh[0], self.entire_mesh[1], self.entire_mesh[2]] 
            colors = [ "#7EC8E3", "#95D5B2", "#F4A5B0"]
            opacities = [1] * len(meshes_free)

            plotter = pv.Plotter()
            plotter.show_grid(color = "white", xtitle='chordwise [m]', ytitle='spanwise [m]', ztitle='Z [m]')

            for m, color, opacity in zip( meshes_entire +  meshes_free, colors + colors, opacities + opacities):
                points_mesh = np.column_stack([m.X.flatten(), m.Y.flatten(), m.Z.flatten()])
                grid = pv.StructuredGrid()
                grid.points = points_mesh
                grid.dimensions = [m.X.shape[1], m.X.shape[0], 1]
                plotter.add_mesh(grid, color=color, show_edges=True, edge_color='black', opacity=opacity)

            plotter.show()

 
    def free_wake_gamma_plot(self,
                             plot_blade: bool = False,
                             limits: list = [10, 125],
                             show_values: bool = False):
        """
        Shows the entire geometry 
        of all vortex rings with 
        the circulation strength 
        values.

        Parameters
        ----------
        plot_blade : bool, optional
            Plot the vortex rings
            representing the blade, 
            by default False
        limits : list, optional
            Legend scale, 
            by default [10, 125]
        show_values : bool, optional
            Show values of each
            vortex ring strength, by default False
        """

        if hasattr(self, 'fw_gamma'):

            scalar_bar_args = dict(
                title='gamma',
                title_font_size=18,
                label_font_size=14,
                color="white",
                fmt="%.2f",
                n_labels=5,
                vertical=True,
                position_x=0.88,
                position_y=0.05,
                width=0.08,
                height=0.85,
            )

            plotter = pv.Plotter()
            plotter.set_background("black")

            def build_grid(mesh, results):
                points = np.column_stack([
                    mesh.X.flatten(),
                    mesh.Y.flatten(),
                    mesh.Z.flatten(),
                ])
                grid = pv.StructuredGrid()
                grid.points = points
                grid.dimensions = [mesh.X.shape[1], mesh.X.shape[0], 1]
                grid.cell_data['gamma'] = results.flatten()
                return grid

            def add_cell_labels(plotter, grid):
                centers = grid.cell_centers()
                labels = [f"{v:.1f}" for v in grid.cell_data['gamma']]
                plotter.add_point_labels(
                    centers,
                    labels,
                    font_size=8,
                    text_color="white",
                    always_visible=True,
                    show_points=False,
                )

            grid = build_grid(self.free_wake[0], self.fw_gamma)
            plotter.add_mesh(grid,
                            scalars='gamma',
                            cmap="rainbow",
                            show_edges=True,
                            edge_color="black",
                            clim=limits,
                            scalar_bar_args=scalar_bar_args)
            if show_values:
                add_cell_labels(plotter, grid)

            if plot_blade:
                grid2 = build_grid(self.blade_meshes[0], self.gamma_results)
                plotter.add_mesh(grid2,
                                scalars='gamma',
                                cmap="rainbow",
                                show_edges=True,
                                edge_color="black",
                                clim=limits,
                                scalar_bar_args=scalar_bar_args)
                if show_values:
                    add_cell_labels(plotter, grid2)

            plotter.add_axes(
                color="white",
                xlabel="X", ylabel="Y", zlabel="Z",
                line_width=3,
            )

            plotter.show_bounds(
                grid=False,
                location="outer",
                ticks="outside",
                font_size=10,
                color="white",
                fmt="%.2f",
                xtitle="chordwise [m]",
                ytitle="spanwise [m]",
                ztitle="Z [m]",
            )

            plotter.show()

        else:
            print('No free wake circulation data in this file')


    def plot_blade_gamma(self, 
                         limits:list = [0,50]):
        """
        Shows the geometry of 
        blade vortex rings with 
        the circulation strength 
        values.

        Parameters
        ----------
        limits : list, optional
            Legend scale, 
            by default [0, 50]
        """
        
        m = self.blade_meshes[0]
        
        results = self.gamma_results

        points = np.column_stack([
            m.X.flatten(), 
            m.Y.flatten(), 
            m.Z.flatten()
        ])
        
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = [m.X.shape[1], m.X.shape[0], 1]
        
        grid.cell_data['gamma'] = results.flatten()
        
        scalar_bar_args = dict(
        title='gamma',
        title_font_size=18,
        label_font_size=14,
        color="white",
        fmt="%.2f",
        n_labels=5,
        vertical=True,
        position_x=0.88,
        position_y=0.05,
        width=0.08,
        height=0.85,
        )

        plotter = pv.Plotter()
        plotter.set_background("black")

        plotter.add_mesh(grid,
                        scalars='gamma',
                        cmap="rainbow",
                        show_edges=True,
                        edge_color="black",
                        clim=limits,
                        scalar_bar_args=scalar_bar_args) 

        plotter.add_axes(
            color="white",
            xlabel="X", ylabel="Y", zlabel="Z",
            line_width=3,
        )

        plotter.show_bounds(
            grid=False,
            location="outer",
            ticks="outside",
            font_size=10,
            color="white",
            fmt="%.2f",
            xtitle="chordwise [m]",
            ytitle="spanwise [m]",
            ztitle="Z [m]",
        )

        plotter.show()

    
    def _get_filaments_mesh(self,
                            type:str = 'spanwise'):
        """Build PyVista line meshes of vortex filaments, colored by circulation strength."""

        blade_mesh = self.blade_meshes[0]
        
        number_panel_spanwise = blade_mesh.shape[1] - 1
        gamma_panels = self.gamma_results.reshape((-1, number_panel_spanwise))
        
        meshes = []

        if type in ('spanwise', 'both'):    
            gamma_spanwise_filaments = np.zeros_like(gamma_panels)
            gamma_spanwise_filaments[0,:] = gamma_panels[0,:]
            gamma_spanwise_filaments[1:,:] = gamma_panels[1:,:] - gamma_panels[:-1,:]


            starting_points = Mesh2D(x_set = blade_mesh.X[:-1,:-1],
                                    y_set = blade_mesh.Y[:-1,:-1],
                                    z_set = blade_mesh.Z[:-1,:-1])
            
            end_points = Mesh2D(x_set = blade_mesh.X[:-1,1:],
                                y_set = blade_mesh.Y[:-1,1:],
                                z_set = blade_mesh.Z[:-1,1:])
            
            N = starting_points.X.size
            points = np.vstack([
                np.column_stack([starting_points.X.ravel(), starting_points.Y.ravel(), starting_points.Z.ravel()]),
                np.column_stack([end_points.X.ravel(), end_points.Y.ravel(), end_points.Z.ravel()])
            ])
            lines = np.column_stack([np.full(N, 2), np.arange(N), np.arange(N) + N]).ravel()

            mesh = pv.PolyData()
            mesh.points = points
            mesh.lines  = lines
            mesh.cell_data["gamma"] = gamma_spanwise_filaments.ravel()
            meshes.append(mesh)

        if type in ('chordwise', 'both'):
            a, b = gamma_panels.shape[0], gamma_panels.shape[1] + 1
            gamma_chordwise_filaments = np.zeros([a, b])
            gamma_chordwise_filaments[:, 0] = gamma_panels[:, 0]
            gamma_chordwise_filaments[:, -1] = gamma_panels[:, -1]
            gamma_chordwise_filaments[:, 1:-1] = gamma_panels[:, 1:] - gamma_panels[:, :-1]
            
            starting_points = Mesh2D(x_set = blade_mesh.X[:-1,:],
                                    y_set = blade_mesh.Y[:-1,:],
                                    z_set = blade_mesh.Z[:-1,:])
            
            end_points = Mesh2D(x_set = blade_mesh.X[1:,:],
                                y_set = blade_mesh.Y[1:,:],
                                z_set = blade_mesh.Z[1:,:])
            
            N = starting_points.X.size
            points = np.vstack([
                np.column_stack([starting_points.X.ravel(), starting_points.Y.ravel(), starting_points.Z.ravel()]),
                np.column_stack([end_points.X.ravel(), end_points.Y.ravel(), end_points.Z.ravel()])
            ])
            lines = np.column_stack([np.full(N, 2), np.arange(N), np.arange(N) + N]).ravel()

            mesh = pv.PolyData()
            mesh.points = points
            mesh.lines  = lines
            mesh.cell_data["gamma"] = gamma_chordwise_filaments.ravel()
            meshes.append(mesh)
        return meshes


    def _get_vectors_mesh(self,
                          type:str = 'lift'):
        """Build PyVista arrow glyphs representing lift or drag force vectors per panel."""

        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                               y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                               z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 
        
        if type == 'lift':
            res = self.lifts_results
        elif type == 'drag':
            res = self.drag_results


        cent = np.column_stack([
            center_points.X.ravel(),
            center_points.Y.ravel(),
            center_points.Z.ravel()
        ])

        direction = np.column_stack([
            res.X.ravel(),
            res.Y.ravel(),
            res.Z.ravel()
        ])

        magnitudes = res.norms.ravel()
        direction_normalized = direction / (magnitudes[:, np.newaxis] + 1e-12)

        mesh = pv.PolyData(cent)
        mesh["vectors"] = direction_normalized
        mesh["magnitude"] = magnitudes

        arrows = mesh.glyph(orient="vectors", scale=False, factor=1)  # scale=False -> wszystkie tej samej dlugosci
        arrows["magnitude"] = np.repeat(magnitudes, arrows.n_cells // len(magnitudes) or 1)

        return arrows

    
    def _get_panel_normals(self):
        """Build PyVista arrow glyphs representing panel normal vectors."""

        blade_mesh = self.blade_meshes[0]
        center_points = blade_mesh.spanwise_midpoints.chordwise_midpoints

        
        cent = np.column_stack([
            center_points.X.ravel(),
            center_points.Y.ravel(),
            center_points.Z.ravel()
        ])

        normals = self.panel_normals
        direction = np.column_stack([
            normals.X.ravel(),
            normals.Y.ravel(),
            normals.Z.ravel()
        ])

        mesh = pv.PolyData(cent)
        mesh["panel_normals"] = direction

        arrows = mesh.glyph(orient="panel_normals", scale=False, factor=1)
    
        return arrows

    
    def _get_free_stream_vectors(self):
        """Build PyVista arrow glyphs representing free stream velocity at collocation points."""

        blade_mesh = self.blade_meshes[0]
        center_points = blade_mesh.spanwise_midpoints.chordwise_midpoints

        
        cent = np.column_stack([
            center_points.X.ravel(),
            center_points.Y.ravel(),
            center_points.Z.ravel()
        ])

        Vcp = self.Vcp

        direction = np.column_stack([
            Vcp.X.ravel(),
            Vcp.Y.ravel(),
            Vcp.Z.ravel()
        ])

        mesh = pv.PolyData(cent)
        mesh["Vcp"] = direction

        arrows = mesh.glyph(orient="Vcp", scale=Vcp, factor=0.05)
    
        return arrows

    
    def _get_velocities_in_lift_locations(self):
        """Build PyVista arrow glyphs representing free stream velocity at lift force locations."""

        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                               y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                               z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 
        
        cent = np.column_stack([
            center_points.X.ravel(),
            center_points.Y.ravel(),
            center_points.Z.ravel()
        ])

        Vil = self.Vil

        direction = np.column_stack([
            Vil.X.ravel(),
            Vil.Y.ravel(),
            Vil.Z.ravel()
        ])

        mesh = pv.PolyData(cent)
        mesh["Vil"] = direction

        arrows = mesh.glyph(orient="Vil", scale=Vil, factor=0.05)
    
        return arrows


    def _get_circulation_vectors(self):
        """Build PyVista arrow glyphs representing filament circulation vector orientation."""

        blade_mesh = self.blade_meshes[0]

        points = np.column_stack([
            blade_mesh.X[:-1,:-1].ravel(),
            blade_mesh.Y[:-1,:-1].ravel(),
            blade_mesh.Z[:-1,:-1].ravel()
        ])

        gamma_vec = self.gamma_vec

        direction = np.column_stack([
            gamma_vec.X.ravel(),
            gamma_vec.Y.ravel(),
            gamma_vec.Z.ravel()
        ])

        mesh = pv.PolyData(points)
        mesh["gamma_vec"] = direction

        arrows = mesh.glyph(orient="gamma_vec", scale=gamma_vec, factor=1)
    
        return arrows

    
    def _get_trailing_edge(self):
        """Build a PyVista line mesh representing the blade trailing edge."""
        TE = self.trailing_edge
        points = np.column_stack([TE.X.ravel(), TE.Y.ravel(), TE.Z.ravel()])

        mesh = pv.PolyData(points)

        lines = np.hstack([[len(points)], np.arange(len(points))])
        mesh.lines = lines
        return mesh


    def plot(self,
             filaments:bool = False,
             filaments_type:str = 'spanwise',
             filaments_limits:list = [0,50],
             vectors_lift:bool = False,
             vectors_drag:bool = False,
             vectors_limits:list = [0,50],
             blade_mesh:bool = False,
             panel_normals:bool = False,
             velocities_in_collocation_points:bool = False,
             velocities_in_lift_locations:bool = False,
             circulation_vectors:bool = False,
             trailing_edge:bool = False
             ):
        """
        Plot different properties on the lifting surface

        Parameters
        ----------
        filaments : bool, optional
            Plot vortex filaments, 
            by default False
        filaments_type : str, optional
            Select type of filaments to plot,
            'spanwise', 'chordwise' or 'both', 
            by default 'spanwise'
        filaments_limits : list, optional
            Define filament legend scale, 
            by default [0,50]
        vectors_lift : bool, optional
            Plot lift vectors, 
            by default False
        vectors_drag : bool, optional
            Plot viscous drag vectors, 
            by default False
        vectors_limits : list, optional
            Define vector legend scale, 
            by default [0,50]
        blade_mesh : bool, optional
            Plot blade vortex ring grid, 
            by default False
        panel_normals : bool, optional
            Plot panel normal vectors, 
            by default False
        velocities_in_collocation_points : bool, optional
            Plot velocity vectors in
            collocation points, 
            by default False
        velocities_in_lift_locations : bool, optional
            Plot velocity vectors in lift locations, 
            by default False
        circulation_vectors : bool, optional
            Plot filament circulation vectors, 
            by default False
        trailing_edge : bool, optional
            Plot trailing edge of the lifting surface, 
            by default False
        """
        
        plotter = pv.Plotter(window_size=[1280, 720])

        plotter.set_background('black')

        second_color = 'white'
        legend_positions = [0.8, 0.05]
        legen_size = [0.08, 0.85]

        if filaments:
            meshes = self._get_filaments_mesh(type=filaments_type)
            scalar_bar_args = dict(
                title="gamma",
                color="white",
                fmt="%.2f",
                n_labels=9,
                vertical=True,
                position_x= legend_positions[0],   # prawa strona
                  position_y= legend_positions[1],
                   width= legen_size[0],
                   height= legen_size[1],
            )
            for i, m in enumerate(meshes):
                plotter.add_mesh(m,
                                 scalars="gamma",
                                 cmap="rainbow",
                                 line_width=2,
                                 n_colors = 8,
                                 clim=filaments_limits,
                                 scalar_bar_args=scalar_bar_args if i == 0 else dict(title="gamma", color="white", fmt="%.2f", n_labels=5))

        if vectors_lift:
            arrows = self._get_vectors_mesh('lift')   
            scalar_bar_args = dict(
            title="Forces [N]",
            color=second_color,
            fmt="%.2f",
            n_labels=9,
            vertical= True,
            position_x= legend_positions[0],   # prawa strona
            position_y= legend_positions[1],
            width= legen_size[0],
            height= legen_size[1],
              )
            plotter.add_mesh(arrows,
                            scalars="magnitude",
                            cmap="rainbow",
                            n_colors = 8,
                            clim=vectors_limits,
                            scalar_bar_args=scalar_bar_args)

        if vectors_drag:
            arrows = self._get_vectors_mesh('drag')   
            scalar_bar_args = dict(
            title="drag",
            color=second_color,
            fmt="%.2f",
            n_labels=9,
            vertical= True,
            position_x= legend_positions[0],   # prawa strona
            position_y= legend_positions[1] - 10,
            width= legen_size[0],
            height= legen_size[1],
              )
            plotter.add_mesh(arrows,
                            scalars="magnitude",
                            cmap="rainbow",
                            n_colors = 8,
                            clim=vectors_limits,
                            scalar_bar_args=scalar_bar_args
                            )

        if blade_mesh:
            mesh = self.blade_meshes[0]
            color = "#FCFCFC"
            opacity = 0.7
            label = 'Blade'

            points_mesh = np.column_stack([mesh.X.flatten(), mesh.Y.flatten(), mesh.Z.flatten()])
            grid = pv.StructuredGrid()
            grid.points = points_mesh
            grid.dimensions = [mesh.X.shape[1], mesh.X.shape[0], 1]
            plotter.add_mesh(grid, color=color, show_edges=True, edge_color='gray', opacity=opacity, label=label,line_width=3)

        if panel_normals:
            normals = self._get_panel_normals()
            plotter.add_mesh(
                normals,
                label = "panel normals",
                line_width=1,
                color = "teal")

        if velocities_in_collocation_points:
            Vcp = self._get_free_stream_vectors()
            plotter.add_mesh(
                Vcp,
                label="V∞",
                line_width = 1,
                color = "blue"
            )

        if velocities_in_lift_locations:
            Vil = self._get_velocities_in_lift_locations()
            plotter.add_mesh(
                Vil,
                label="V in lift location",
                line_width = 1,
                color = "blue"
            )

        if circulation_vectors:
            gamma_vec = self._get_circulation_vectors()
            plotter.add_mesh(
                gamma_vec,
                label="vortex orientation",
                line_width = 1,
                color = "red"
            )

        if trailing_edge:
            line = self._get_trailing_edge()
            plotter.add_mesh(
                line,
                color = 'red',
                line_width = 3,
                label = 'trailing edge'
            )

        plotter.add_axes(color=second_color)
        plotter.show_bounds(grid=False, 
                            location="outer", 
                            ticks="outside",
                            font_size=10, 
                            color=second_color, 
                            fmt="%.2f",
                            xtitle="chordwise [m]", 
                            ytitle="spanwise [m]", 
                            ztitle="Z [m]")

        plotter.show()


    @property
    def normal_forces(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Normal forces per spanwise section
            without the viscous drag [N/m]
        """
        lift = self.lifts_results
        res  = np.sum(lift.Z, axis = 0) / self.panel_widths
        return res


    @property
    def normal_forces_wdrag(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Normal forces per spanwise section
            with the viscous drag [N/m]
        """
        lift = self.lifts_results
        drag = self.drag_results
        res = (np.sum(lift.Z, axis = 0) + np.sum(drag.Z, axis = 0)) /  self.panel_widths
        return res


    @property
    def tangent_forces(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Tangent forces per spanwise section
            without the viscous drag [N/m]
        """
        lift = self.lifts_results
        
        F_x = lift.X
        F_y = lift.Y

        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                                y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                                z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 

        x = center_points.X
        y = center_points.Y

        F_tangent = (F_x * (-y) + F_y * x) / np.sqrt(x**2 + y**2)  

        res = np.sum(F_tangent, axis = 0)  / self.panel_widths
        return res


    @property
    def tangent_forces_wdrag(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Tangent forces per spanwise section
            with the viscous drag [N/m]
        """
        lift = self.lifts_results
        drag = self.drag_results
        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                                y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                                z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 

        x = center_points.X
        y = center_points.Y

        F_x = lift.X + drag.X
        F_y = lift.Y + drag.Y

        F_tangent = (F_x * (-y) + F_y * x) / np.sqrt(x**2 + y**2)  

        res = np.sum(F_tangent, axis = 0)  / self.panel_widths

        return res


    @property
    def force_locations(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Spawise force locations
        """
        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                                y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                                z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 

        spanwise_locations = np.mean(center_points.Y, axis = 0)
        return spanwise_locations


    def plot_forces2D(self,
                      force_type:str = 'normal',
                      include_drag:bool = False):
        """
        Plot spanwise force distribution

        Parameters
        ----------
        force_type : str, optional
            Normal or tangent, 
            by default 'normal'
        include_drag : bool, optional
            Include viscous drag in 
            force calculation, 
            by default False
        """

        lift = self.lifts_results
        if include_drag:
            drag = self.drag_results

        blade_mesh = self.blade_meshes[0]

        center_points = Mesh2D(x_set = blade_mesh.spanwise_midpoints.X[:-1,:],
                                y_set = blade_mesh.spanwise_midpoints.Y[:-1,:],
                                z_set = blade_mesh.spanwise_midpoints.Z[:-1,:]) 

        spanwise_locations = np.mean(center_points.Y, axis = 0)
    
        import matplotlib.pyplot as plt

        if force_type == 'normal':
            res  = np.sum(lift.Z, axis = 0) / self.panel_widths
            if include_drag:
                res = res + np.sum(drag.Z, axis = 0) / self.panel_widths
        
        elif force_type == 'tangent':
            F_x = lift.X
            F_y = lift.Y

            if include_drag:
                F_x = F_x + drag.X
                F_y = F_y + drag.Y

            x = center_points.X
            y = center_points.Y

            F_tangent = (F_x * (-y) + F_y * x) / np.sqrt(x**2 + y**2)  

            res = np.sum(F_tangent, axis = 0)  / self.panel_widths

        else:
            raise ValueError('wrong force type input')
        
        plt.plot(spanwise_locations, res, '.-')
        plt.title(f"Wind speed {self.V:.2f} [m/s]")
        plt.xlabel('Rotor radius [m]')
        plt.ylabel(force_type + ' forces [N/m]')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


    def thrust_per_blade(self,
                         include_drag:bool = False) -> float:
        """
        Computes thrust force per rotor blade

        Parameters
        ----------
        include_drag : bool, optional
            Include viscous drag into computation, 
            by default False

        Returns
        -------
        float
            Thrust per blade [N]
        """
        if include_drag:
            return np.sum(self.lifts_results.Z) + np.sum(self.drag_results.Z) 
        else:
            return np.sum(self.lifts_results.Z)

    
    def rotational_moment_per_blade(self,
                                    include_drag:bool =  False) -> float:
        """
        Computes rotational moment per rotor blade

        Parameters
        ----------
        include_drag : bool, optional
            Include viscous drag into computation, 
            by default False

        Returns
        -------
        float
            Rotatational moment per blade [Nm]
        """
        blade_mesh = self.blade_meshes[0]
        x = blade_mesh.spanwise_midpoints.X[:-1,:].ravel(order = 'C')
        y = blade_mesh.spanwise_midpoints.Y[:-1,:].ravel(order = 'C')

        if include_drag:
            Fx = self.lifts_results.X.ravel(order = 'C') + self.drag_results.X.ravel(order = 'C')
            Fy = self.lifts_results.Y.ravel(order = 'C') + self.drag_results.Y.ravel(order = 'C')
        else:
            Fx = self.lifts_results.X.ravel(order = 'C')
            Fy = self.lifts_results.Y.ravel(order = 'C')
            
        Mz1 = np.sum(x * Fy)
        Mz2 = np.sum(-y * Fx)

        return Mz1 + Mz2 

    
    def power(self,
              include_drag:bool = False) -> float:
        """
        Computes rotor power

        Parameters
        ----------
        include_drag : bool, optional
            Include viscous drag into computation, 
            by default False

        Returns
        -------
        float
            Rotor power [W]
        """
        return self.omega * self.blades * self.rotational_moment_per_blade(include_drag)


    def get_CT(self,
               rotor_plane_radius:float,
               include_drag:bool = False) -> float:
        """
        Computes thrust coefficient

        Parameters
        ----------
        rotor_plane_radius : float
            Radius of the area swept by the rotor [m]
        include_drag : bool, optional
            Include the viscous drag into computation, 
            by default False

        Returns
        -------
        float
            Thrust coefficient [-]
        """
        rotor_area =  np.pi * rotor_plane_radius ** 2
        return self.thrust_per_blade(include_drag) * self.blades / (0.5 * self.air_density * rotor_area * self.V ** 2)

    
    def get_CP(self,
               rotor_plane_radius:float,
               include_drag:bool = False) -> float:
        """
        Computes power coefficient

        Parameters
        ----------
        rotor_plane_radius : float
            Radius of the area swept by the rotor [m]
        include_drag : bool, optional
            Include the viscous drag into computation, 
            by default False
            
        Returns
        -------
        float
            Power coefficient [-]
        """
        rotor_area =  np.pi * rotor_plane_radius ** 2
        return self.power(include_drag) / (0.5 * self.air_density * rotor_area * self.V ** 3)
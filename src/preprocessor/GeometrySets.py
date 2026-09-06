import numpy as np


def rotation_matrix_by_angle(angle:float):
    """
    Creates rotation matrix around the second dimension axis (Y)

    Parameters
    ----------
    angle : float
        Rotation angle [rad] 

    Returns
    -------
        Rotation matrix based on the particular angle
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rotation_matrix_by_angleXY(angle:float):
    """
    Creates rotation matrix around the second dimension axis (Z)
    
    Parameters
    ----------
    angle : float
        Rotation angle [rad]

    Returns
    -------
        Rotation matrix based on the particular angle
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0 ],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])


class Vector:
    """
    Vector object possible to be rotated by angle around the Y axis,
    rotated by the rotation matrix or projected on plane.
    """
    def __init__(self, 
                 vector:list | np.ndarray):
        """
        Vector initialization

        Parameters
        ----------
        vector : list | np.ndarray
            Vector values
        """
        self.vector = np.array(vector)
        

    def __array__(self, dtype = None) -> np.ndarray:
        """
        Parameters
        ----------
        dtype : _type_, optional
            data type, 
            by default None

        Returns
        -------
        np.ndarray
            Array of the vector coordinates
        """
        return np.asarray(self.vector, dtype = dtype)
    

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Vector: {vector values}
        """
        return f"Vector:{self.vector!r})"
    

    def rotated_by_angle(self, 
                         angle:float):
        """
        Rotates vector by defined angle around the Y axis 
        Y axis defines the spanwise direction in the classic 3D analysis,
        an input angle corresponds then to the angle of attack definition

        Parameters
        ----------
        angle : float
            Rotation angle [rad]

        Returns
        -------
        Vector
            Rotated vector
        """
        rotation_matrix = rotation_matrix_by_angle(angle = angle)
        new_vector = rotation_matrix @ self.vector
        return self.__class__(vector = new_vector)

    
    def rotated_by_matrix(self, 
                          matrix:np.ndarray):
        """
        Rotates vector by defined rotation matrix

        Parameters
        ----------
        matrix : np.ndarray
            Rotation matrix

        Returns
        -------
        Vector
            Rotated vector
        """
        new_vector = matrix @ self.vector
        return self.__class__(vector = new_vector)


    def projected_on_plane(self, 
                           plane_vector:np.ndarray):
        """
        Parameters
        ----------
        plane_vector : np.ndarray
            Vector normal to the projection plane

        Returns
        -------
        np.ndarray  
            Projected vector
        """
        projection = self.vector @ plane_vector / (np.linalg.norm(plane_vector) ** 2)
        component_along = projection * plane_vector
        component_parpendicular = self.vector - component_along
        return self.__class__(vector = component_parpendicular)
    

    @property
    def normalized(self):
        """
        Returns
        -------
        Vector
            Normalized vector
        """
        return Vector(vector = self.vector / np.linalg.norm(self.vector))


    @property
    def norm(self) -> float:
        """
        Returns
        -------
        float
            Vector norm
        """
        return np.linalg.norm(self.vector)


class GeometrySet:
    """
    Geometry representation of the object in the 3D field,
    the base class for classes such as:
        - VectorSet
        - Beam
        - CamberLine
        - Mesh2D
    """
    def __init__(self, 
                 x_set:np.ndarray = None, 
                 y_set:np.ndarray = None,
                 z_set:np.ndarray = None, 
                 one_vector_input:bool = False,
                 one_vector = None,
                 blank:bool = False,
                 size:int = None,
                 ):
        """
        Initialize a geometry object

        Parameters
        ----------
        x_set : np.ndarray
            X coordinate for every node in the set
        y_set : np.ndarray
            Y coordinate for every node in the set
        z_set : np.ndarray
            Z coordinate for every node in the set
        one_vector_input : bool
            Flag for generating input by one vector
        one_vector : 
            Input with the one vector: [x1, y1, z1, x2, y2, z2, ...]
        blank : bool
            Flag for generating vector with zeros
        size : int
            Size of the vector

        Raises
        ------
        ValueError
            size cannot be None
        ValueError
            one_vector cannot be None 
        ValueError
            one_vector length must be divisible by 3
        ValueError
            x_set, y_set, z_set all cannot be None
        ValueError
            x_set, y_set, z_set must have the same length
        """

        # defines geometry set with defined shape containing only zeros
        if blank:
            if size is None:
                raise ValueError("size required when blank = True")
            else:    
                self.X = np.zeros(size)
                self.Y = np.zeros(size)
                self.Z = np.zeros(size)
        # defines geometry set with flatten array, ex: [X1, Y1, Z1, X2, Y2, Z2...]
        elif one_vector_input:
            one_vector = np.array(one_vector).flatten() 
            if one_vector is None:
                raise ValueError("one_vector required when one_vector_input = True")
            vector_input = np.array(one_vector).flatten()
            if len(vector_input) % 3 !=0:
                raise ValueError("vector length must be divisible by 3")
            else:
                self.X = vector_input[0::3]
                self.Y = vector_input[1::3]
                self.Z = vector_input[2::3]
        # defines geometry set with the three separated coordinate sets
        else:
            if x_set is None or y_set is None or z_set is None:
                raise ValueError("x_set, y_set, z_set all required when one_vector_input=False")
            if not len(np.array(x_set)) == len(np.array(y_set)) == len(np.array(z_set)):
                raise ValueError("x, y, z must have the same length")
            else:
                self.X = np.array(x_set) if not isinstance(x_set, np.ndarray) else x_set
                self.Y = np.array(y_set) if not isinstance(y_set, np.ndarray) else y_set
                self.Z = np.array(z_set) if not isinstance(z_set, np.ndarray) else z_set


    def get_node(self, index) -> np.ndarray:
        """ 
        Returns the node coordinates

        Parameters
        ----------
        index : int
            index number of the considered node in the set

        Returns
        -------
        np.ndarray
            array of node index
        """
        return np.array([self.X[index], self.Y[index], self.Z[index]])


    @property
    def shape(self) -> tuple:
        """
        Returns
        -------
        tuple
            Shape of the geometry set
        """
        return(self.X.shape)


    @property
    def as_vector(self) -> np.ndarray:
        """
        Transforms a GeometrySet into np.array matrix

        Returns
        -------
        np.ndarray
            One matrx representing all of the coordinates in the GeometrySet
        """
        if self.X.ndim == 2:
            vec = np.column_stack([
                self.X.flatten(order='C'),
                self.Y.flatten(order='C'),
                self.Z.flatten(order='C')
            ]).flatten()
            return vec
        else:
            return np.column_stack([self.X, self.Y, self.Z]).flatten()


class VectorSet(GeometrySet):
    """
    Stores the vector field representation,
    which can be normalized, converted into
    vector nomrms, rotated by angle around the Y axis
    or by the rotation vector, projected on defined planes,
    multiplied by scalar, multiplied by single or particular vector 
    for each of the vectors in the VectorSet
    """
    def __init__(self, *args, **kwargs):
        """
        Initiate the vector set,
        for more details check the 
        GeometrySet initialization
        """
        super().__init__(*args, **kwargs)


    @property
    def as_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            VectorSet in matrix form stacked on axis = 1
        """
        return np.stack((self.X, self.Y, self.Z), axis = 1)


    @property
    def as_matrix_nx3(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            VectorSet in matrix form stacked on axis -1 (..., 3),
            Preferable for cross product operation.
        """
        return np.stack([self.X, self.Y, self.Z], axis=-1)


    @property
    def norms(self) -> np.ndarray:
        """ 
        Returns
        -------
        np.ndarray
            vector norms
        """
        return np.sqrt(self.X**2 + self.Y**2 + self.Z**2)


    @property
    def normalized(self):
        """
        Normalises the VectorSet

        Returns
        -------
        VectorSet
            Normalized vector set
        """
        return self.__class__(x_set=self.X/self.norms, y_set=self.Y/self.norms, z_set=self.Z/self.norms)   
    

    def __mul__(self,
                scalar:float):
        """
        Multiply VectorSet by
        the defined Scalar

        Parameters
        ----------
        scalar : float
            Multiplification factor

        Returns
        -------
        VectorSet
            Scaled VectorSet
        """
        return self.__class__(
            x_set = self.X * scalar,
            y_set = self.Y * scalar,
            z_set = self.Z * scalar,
        )

    
    def __rmul__(self,
                 scalar:float):
        """
        Multiply VectorSet by
        the defined Scalar

        Parameters
        ----------
        scalar : float
            Multiplification factor

        Returns
        -------
        VectorSet
            Scaled VectorSet
        """
        return self.__mul__(scalar)


    def get_vector(self,index):
        """ 
        Returns the vectors for an index input

        Parameters
        ----------
        index : int
            index number of the considered vector in the set

        Returns
        -------
        Vector
            Vector of an input index
        """
        return Vector(vector = np.array([self.X[index], self.Y[index], self.Z[index]]))   


    def rotated_by_matrix(self,
                          matrix:np.ndarray,
                          big_matrix:bool = False, 
                          mesh:bool = False):
        """
        Rotates the set by the rotation matrix

        Parameters
        ----------
        matrix : np.ndarray
            Rotation matrix
        big_matrix : bool, optional
            True if the matrix is not 3x3 but 
            has the size of the entire VectorSet, 
            by default False
        mesh : bool, optional
            Used when rotating multi-blade mesh geometry 
            (see ComputationPanels.compute_influence_matrix).
            by default False

        Returns
        -------
        VectorSet
            Rotated VectorSet
        """
        if not big_matrix:                
            from scipy import sparse
            if mesh is not True:
                rotation_matrix = sparse.kron(sparse.eye(self.shape[0]), matrix)
            else:
                rotation_matrix = sparse.kron(sparse.eye(self.as_vector.size // 3), matrix)

            new_vectors = rotation_matrix @ self.as_vector
            return self.__class__(one_vector_input = True,
                                one_vector = new_vectors)
        else:
            new_vectors = matrix @ self.as_vector
            return self.__class__(one_vector_input = True,
                                  one_vector = new_vectors)


    def rotated_by_angle(self, 
                         angle:float, 
                         XY = False, 
                         mesh = False):
        """
        Rotates the set by angle

        Parameters
        ----------
        angle : float
            Rotation angle [rad]
        XY : bool, optional
            If True rotates the set around Z axis,
            by default False which means the rotation 
            around the Y axis
        mesh : bool, optional
            Used when rotating multi-blade mesh geometry 
            (see ComputationPanels.compute_influence_matrix).
            by default False

        Returns
        -------
        VectorSet
            Rotated VectorSet
        """
        if XY:
            R = rotation_matrix_by_angleXY(angle = angle)
        else:
            R = rotation_matrix_by_angle(angle = angle)
        
        if mesh:
            return self.rotated_by_matrix(R, mesh = True)
        else:
            return self.rotated_by_matrix(R)
    

    def cross_with_single(self, 
                          vec:Vector | np.ndarray, 
                          flatten:bool = False):
        """
        Gives cross product of the VectorSet with
        single vector

        Parameters
        ----------
        vec : Vector | np.ndarray
            Vector description
        flatten : bool, optional
            Uses the as_matrix_nx3 for
            the 2D mesh option, 
            by default False

        Returns
        -------
        VectorSet
            Set of cross products
        """
        
        v = vec.vector if isinstance(vec, Vector) else np.array(vec) # (3,)

        if flatten:
            vecs_matrix = self.as_matrix_nx3  # (n*m, 3)
            crosses = np.cross(v, vecs_matrix)  # (n*m, 3)
            # Reshape to the original shape
            return self.__class__(
                x_set=crosses[:, :, 0],
                y_set=crosses[:, :, 1],
                z_set=crosses[:, :, 2])

        else:
            vecs_matrix = self.as_matrix # shape (3, n)
            crosses = np.cross(v, vecs_matrix).T # (n, 3).T = (3, n)
            return self.__class__(x_set = crosses[0], y_set = crosses[1], z_set = crosses[2])
    

    def cross_with_set(self, 
                       other_set):
        """
        Computes the cross product between two VectorSets.

        Parameters
        ----------
        other_set : VectorSet
            Second factor VectorSet

        Returns
        -------
        VectorSet
            Result of the cross product
        """
        crosses = np.cross(self.as_matrix_nx3, other_set.as_matrix_nx3)  # (..., 3)
        crosses = np.moveaxis(crosses, -1, 0)  # (3, m, n) or (3, n) depends on the input
        return self.__class__(x_set = crosses[0], y_set = crosses[1], z_set = crosses[2])


    def projected_on_planes(self, 
                            plane_normals):
        """
        Projects the VectorSet on the
        plane defined by the normals.

        Parameters
        ----------
        plane_normals : VectorSet
            Vectors normals to the defined planes.

        Returns
        -------
        VectorSet
            Vector projected of the respective planes
        """
        dots = (self.X * plane_normals.X +
                self.Y * plane_normals.Y +
                self.Z * plane_normals.Z)
        
        norms2 = (plane_normals.X**2 + plane_normals.Y**2 + plane_normals.Z**2)
        projections = dots / norms2

        # Components along normals
        comp_x = projections * plane_normals.X
        comp_y = projections * plane_normals.Y
        comp_z = projections * plane_normals.Z

        # Substract to get parpendicular components
        return self.__class__(x_set = self.X - comp_x,
                              y_set = self.Y - comp_y,
                              z_set = self.Z - comp_z)


class Beam(GeometrySet):
    """
    Beam geometric representation, supporting rotation, coordinate
    system transforms, and conversion between global/local frames.
    """
    def __init__(self, *args, **kwargs):         
        """
        Initiate the beam,
        for more details check the 
        GeometrySet initialization
        """
        super().__init__(*args, **kwargs)


    @property 
    def _distance_between_sections(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Distances between the beam nodes
        """
        return np.abs(np.diff(self.Y))


    def setup_global_local_transform(self) -> np.ndarray:
        """
        Setup the rotation matrix to trasform the coordinates
        into the local coordinate system of beam and in reverse

        Returns
        -------
        np.ndarray
            Rotation matrix from global to local CS
        np.ndarray
            Rotation matrix from local to global CS
        """
        e1 = np.array([1,0,0])  
        e2 = np.array([0, self.Y[1]-self.Y[0], self.Z[1]-self.Z[0]])
        e2 = e2/np.linalg.norm(e2)
        e3 = np.cross(e1,e2)
        R_g2l = np.vstack([e1, e2, e3])
        R_l2g = R_g2l.T
        return R_g2l, R_l2g
    

    def normal_vectors(self, 
                       plane:str) -> VectorSet:
        """
        Calculates the normal vectors to the Beam

        Parameters
        ----------
        plane : str
            XY, YZ, XZ, defines the plane in 
            wich normal vectors are calculated

        Returns
        -------
        VectorSet
            Vectors normal to the Beam in defined plane
        """
        dx, dy, dz = np.gradient(self.X), np.gradient(self.Y), np.gradient(self.Z)
        
        match plane:
            case "XY": vecs = [-dy, dx, np.zeros_like(dz)]
            case "YZ": vecs = [np.zeros_like(dx), -dz, dy]
            case "XZ": vecs = [-dz, np.zeros_like(dy), dx]
            case _: raise ValueError('in_plane variable must be provided (XY, YZ, or XZ)')
        
        norm = np.linalg.norm(vecs, axis=0)
        return VectorSet(x_set = vecs[0]/norm, y_set = vecs[1]/norm, z_set = vecs[2]/norm)


    def rotated_by_matrix(self,
                          matrix:np.ndarray,
                          big_matrix:bool = False):
        """
        Rotates the beam by the rotation matrix

        Parameters
        ----------
        matrix : np.ndarray
            Rotation matrix
        big_matrix : bool, optional
            True if the matrix is not 3x3 but 
            has the size of the entire Beam, 
            by default False

        Returns
        -------
        Beam
            Rotated beam
        """
        if not big_matrix:                
            from scipy import sparse
            rotation_matrix = sparse.kron(sparse.eye(self.shape[0]), matrix)
            new_vectors = rotation_matrix @ self.as_vector
            return self.__class__(one_vector_input = True,
                                one_vector = new_vectors)
        else:
            new_vectors = matrix @ self.as_vector
            return self.__class__(one_vector_input = True,
                                  one_vector = new_vectors)
    

    def in_local_CS(self):
        """
        Returns
        -------
        Beam
            Moves the Beam to its local coordinates 
        """
        R_g2l = self.setup_global_local_transform()[0]
        root_node = self.get_node(index = 0)
        current_vector = self.as_vector.reshape(-1,1)
        root_vector = np.tile(root_node.reshape(-1,1), (self.shape[0], 1)) # add error in terms of 2d matrix for coordinates
        return (self.__class__(one_vector_input = True,
                               one_vector = current_vector - root_vector).
                               rotated_by_matrix(R_g2l))


    def moved_to_(self, 
                  coordinate_system:str ,
                  reference_node:np.ndarray, 
                  rotation_matrix:np.ndarray):
        """
        Moves beam to defined 
        coordiante system

        Parameters
        ----------
        coordinate_system : str
            global or local
        reference_node : np.ndarray
            The beginning node of the movement
        rotation_matrix : np.ndarray
            Rotation matrix

        Returns
        -------
        Beam
            Beam rotated to the desired CS
        """
        current_vector = self.as_vector.reshape(-1,1)
        rotated_vector =(self.__class__(one_vector_input = True,
                                        one_vector = current_vector).
                                        rotated_by_matrix(rotation_matrix))
    
        root_vector = np.tile(reference_node,self.shape[0])
        
        if coordinate_system == 'global':
            new_vector = rotated_vector.as_vector + root_vector
        elif coordinate_system =='local':
            new_vector = rotated_vector.as_vector - root_vector
        else:
            raise ValueError("destination coordinate system must be defined as: global or local (str)")
        
        return self.__class__(one_vector_input = True,
                              one_vector = new_vector)


class CamberLine(GeometrySet):
    """
    Camber line surface definition
    """
    def __init__(self, *args, **kwargs):
        """
        Camberline surface initialization
        """
        super().__init__(*args, **kwargs)

    @property
    def as_matrix(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Matrix np.ndarray representation
        """
        return np.stack((self.X, self.Z), axis=1).reshape(-1, len(self.X[0,:]))


class Mesh2D(GeometrySet):
    """
    2D structured geometry grid (chordwise x spanwise), supporting
    mesh specific operations like chord fraction locations, midpoints,
    reshaping, and splitting.
    """
    def __init__(self, *args, **kwargs):
        """
        Initiate the mesh,
        for more details check the
        GeometrySet initialization
        """
        super().__init__(*args, **kwargs)


    @property
    def size(self) -> int:
        """
        Returns
        -------
        int
            Number of the nodes defined in the Mesh2D
        """
        return self.X.size


    @property
    def trailing_edge(self) -> GeometrySet:
        """
        Returns
        -------
        GeometrySet
            Trailing edge geometry representation
        """
        return GeometrySet(x_set = self.X[-1,:],
                           y_set = self.Y[-1,:],
                           z_set = self.Z[-1,:])


    @property
    def spanwise_vector_set(self) -> VectorSet:
        """
        Returns
        -------
        VectorSet
            Set of the vectors between the boundary Mesh2D elements in the spanwise direction
        """
        return VectorSet(x_set = self.X[:,1:] - self.X[:,:-1],
                         y_set = self.Y[:,1:] - self.Y[:,:-1],
                         z_set = self.Z[:,1:] - self.Z[:,:-1])
    

    @property
    def quarter_chords(self):
        """
        Returns
        -------
        Mesh2D
            Quarter chord locations
        """
        return Mesh2D(x_set = self.X[:-1] * 0.75 + self.X[1:] * 0.25,
                      y_set = self.Y[:-1] * 0.75 + self.Y[1:] * 0.25,
                      z_set = self.Z[:-1] * 0.75 + self.Z[1:] * 0.25)


    @property
    def three_quarter_chords(self):
        """
        Returns
        -------
        Mesh2D
            Three quarter chord locations
        """
        return Mesh2D(x_set = self.X[:-1] * 0.25 + self.X[1:] * 0.75,
                      y_set = self.Y[:-1] * 0.25 + self.Y[1:] * 0.75,
                      z_set = self.Z[:-1] * 0.25 + self.Z[1:] * 0.75)
    

    @property
    def spanwise_midpoints(self):
        """
        Returns
        -------
        Mesh2D
            Spanwise midpoint locations
        """
        return Mesh2D(x_set = (self.X[:,:-1] + self.X[:,1:]) / 2,
                      y_set = (self.Y[:,:-1] + self.Y[:,1:]) / 2,
                      z_set = (self.Z[:,:-1] + self.Z[:,1:]) / 2)
    
    @property
    def chordwise_midpoints(self):
        """
        Returns
        -------
        Mesh2D
            Spanwise midpoint locations
        """
    
        return Mesh2D(x_set = (self.X[:-1,:] + self.X[1:,:]) / 2,
                      y_set = (self.Y[:-1,:] + self.Y[1:,:]) / 2,
                      z_set = (self.Z[:-1,:] + self.Z[1:,:]) / 2)


    def reshape_set(self, 
                    new_shape:np.ndarray):
        """
        Reshapes the geometry set in the C order

        Parameters
        ----------
        new_shape : np.ndarray
            New shape of the geometry set
        """
        self.X = self.X.reshape(new_shape, order='C')
        self.Y = self.Y.reshape(new_shape, order='C')
        self.Z = self.Z.reshape(new_shape, order='C')


    def split_set_by_row(self, 
                         row_number:int):
        """
        Splits the Mesh2D in two Mesh2D based on the row number

        Parameters
        ----------
        row_number : int
            Number of the row in the intial Mesh2D

        Returns
        -------
        Mesh2D
            Object with the rows below and with the defined input row number
        Mesh2D
            Object with the rows above and with the defined input row number
        """
        
        up_product = Mesh2D(x_set = self.X[:row_number,:],
                            y_set = self.Y[:row_number,:],
                            z_set = self.Z[:row_number,:])
        down_product = Mesh2D(x_set = self.X[row_number -1:,:],
                              y_set = self.Y[row_number -1:,:],
                              z_set = self.Z[row_number -1:,:])
        return up_product, down_product
        

class PanelSet:
    """
    Panel representation built from a Mesh2D, providing panel level
    geometric properties such as area and normal vectors.
    """
    def __init__(self, 
                 mesh_set:Mesh2D):
        """
        Initiates the panel set

        Parameters
        ----------
        mesh_set : Mesh2D
            Mesh 2D on which the panel set is created
        """
        self.panel_grid = mesh_set


    @property
    def areas(self) -> np.ndarray:
        """
        Calculates the panel areas based on the diagonals cross products

        Returns
        -------
        np.ndarray
            Panel areas
        """
        first_diagonals = VectorSet(x_set = self.panel_grid.X[:-1,:-1] - self.panel_grid.X[1:,1:],
                                    y_set = self.panel_grid.Y[:-1,:-1] - self.panel_grid.Y[1:,1:],
                                    z_set = self.panel_grid.Z[:-1,:-1] - self.panel_grid.Z[1:,1:])
        
        second_diagonals = VectorSet(x_set = self.panel_grid.X[:-1,1:] - self.panel_grid.X[1:,:-1],
                                     y_set = self.panel_grid.Y[:-1,1:] - self.panel_grid.Y[1:,:-1],
                                     z_set = self.panel_grid.Z[:-1,1:] - self.panel_grid.Z[1:,:-1])

        return first_diagonals.cross_with_set(other_set = second_diagonals).norms * 0.5
    

    @property
    def normals(self) -> VectorSet:
        """
        Calculates the panel normals

        Returns
        -------
        VectorSet
            Panel normal vectors, computed from the cross product of panel diagonals.
        """
        collocation_points = self.panel_grid.three_quarter_chords.spanwise_midpoints

        first_diagonals = VectorSet(x_set = self.panel_grid.quarter_chords.X[:,:-1] - collocation_points.X,
                                    y_set = self.panel_grid.quarter_chords.Y[:,:-1] - collocation_points.Y,
                                    z_set = self.panel_grid.quarter_chords.Z[:,:-1] - collocation_points.Z)

        second_diagonals = VectorSet(x_set = self.panel_grid.quarter_chords.X[:,1:] - collocation_points.X,
                                     y_set = self.panel_grid.quarter_chords.Y[:,1:] - collocation_points.Y,
                                     z_set = self.panel_grid.quarter_chords.Z[:,1:] - collocation_points.Z)

        return second_diagonals.cross_with_set(other_set = first_diagonals).normalized
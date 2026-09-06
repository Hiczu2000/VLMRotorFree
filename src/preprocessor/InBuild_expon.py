#%%
import numpy as np

def skew_matrix(vector, vector_derivative = None):
    x = vector
    skew_matrix = np.array([[0, -x[2], x[1]],
                            [x[2], 0, -x[0]],
                            [-x[1], x[0], 0]])
    
    if vector_derivative is None:
        return skew_matrix
    
    else:
        dx = vector_derivative
        n = dx.shape[1] if dx.ndim > 1 else 1
        
        der_skew = np.zeros((3, 3, n))
        der_skew[0, 1, :] = -dx[2, :]  
        der_skew[0, 2, :] =  dx[1, :]  
        der_skew[1, 0, :] =  dx[2, :]  
        der_skew[1, 2, :] = -dx[0, :]  
        der_skew[2, 0, :] = -dx[1, :]
        der_skew[2, 1, :] =  dx[0, :]

        return skew_matrix,der_skew

def rotation_matrix_rodrigues(vector, vector_derivative = None):
    
    I = np.eye(3)
    t = vector.flatten()

    # vector norm
    al = np.linalg.norm(t)
    # skew matrix
    Rsk = skew_matrix(vector = t)
    
    # rotation matrix
    if al < 1e-15:
        R = I.copy()
    else:
        Rsk2 = Rsk @ Rsk
        R = I + (np.sin(al)/al) * Rsk + 0.5 * (np.sin(al/2)/(al/2)) ** 2 * Rsk2
    if vector_derivative is None:
        return R
    else:
        # first derivative calc
        if al < 1e-15:
            dR = dRsk
        else:    
            dt = vector_derivative
            Rsk, dRsk = skew_matrix(vector = t, vector_derivative = dt)
            n = dt.shape[1]

            # norm derivative
            dal = (1 / (2 * al)) * (2 * t[0] * dt[0,:] + 2 * t[1] * dt[1,:] + 2 * t[2] * dt[2,:])
            dal = dal.reshape(1,n) 

            # współczynniki
            c1 = (1/al**2) * (np.cos(al) * al - np.sin(al)) * dal # [1,n]
            c2 = 4*np.sin(al/2)/al**3 * (0.5 * np.cos(al/2) * al - np.sin(al/2)) * dal # [1,n]

            # jak to działa
            Rskc1 = Rsk[:,:,None] * c1[None,None,:]  # [3,3,1] * [1,1,n] = [3,3,n]
            Rskc2 = Rsk2[:,:,None] * c2[None,None,:]

            # RdRdRRp[i] = Rsk @ dRsk[:,:,i] + dRsk[:,:,i] @ Rsk
            RdRdRR = np.zeros((3,3,n))
            for i in range(n):
                RdRdRR[:,:,i] = Rsk @ dRsk[:,:,i] + dRsk[:,:,i] @ Rsk

            dR = (Rskc1 + (np.sin(al)/al) * dRsk + Rskc2 + 2 * (np.sin(al/2)/al)**2 * RdRdRR )
        # second derivative calc
        if al < 1e-15:
            d2R = np.zeros((3, 3, n, n))
        else:
            d2R = np.zeros((3,3, n, n))

            for i in range(n):
                for j in range(n):
                    # d2al second derivative of norm
                    term1 = -(1/al**3) * (t[0]*dt[0,j] + t[1]*dt[1,j] + t[2]*dt[2,j]) * \
                                         (t[0]*dt[0,i] + t[1]*dt[1,i] + t[2]*dt[2,i])
                    term2 = (1/al) * (dt[0,j]*dt[0,i] + dt[1,j]*dt[1,i] + dt[2,j]*dt[2,i])
                    d2al = term1 + term2
                    # coefficients
                    c1 = (1/al**2) * (np.cos(al)*al - np.sin(al))
                    c2 = (al**2*(-al*np.sin(al)) - (np.cos(al)*al - np.sin(al))*2*al) / al**4
                    c3 = 4*(al**3*0.5*np.cos(al/2) - np.sin(al/2)*3*al**2)/al**6 * (al/2*np.cos(al/2) - np.sin(al/2))
                    c4 = 4*np.sin(al/2)/al**3 * (-0.5*al/2*np.sin(al/2))
                    c5 = 4*np.sin(al/2)/al**3 * (0.5*np.cos(al/2)*al - np.sin(al/2))
                    c6 = 0.5 * (np.sin(al/2)/(al/2))**2

                    d2R[:,:,i,j] = (c1 * Rsk * d2al +
                                    c1 * dRsk[:,:,j] * dal[0,i] +
                                    c2 * Rsk * dal[0,i] * dal[0,j] +
                                    c1 * dal[0,j] * dRsk[:,:,i] +
                                    c3 * Rsk2 * dal[0,i] * dal[0,j] +
                                    c4 * Rsk2 * dal[0,i] * dal[0,j] +
                                    c5 * dal[0,i] * (Rsk @ dRsk[:,:,j] + dRsk[:,:,j] @ Rsk) +
                                    c5 * Rsk2 * d2al +
                                    c5 * dal[0,j] * (Rsk @ dRsk[:,:,i] + dRsk[:,:,i] @ Rsk) +
                                    c6 * (dRsk[:,:,i] @ dRsk[:,:,j] + dRsk[:,:,j] @ dRsk[:,:,i]))
            return R, dR, d2R
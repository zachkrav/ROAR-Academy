import numpy as np

v = np.array([2., 2., 4.])

e0 = np.array([1., 0., 0.])
e1 = np.array([0., 1., 0.])
e2 = np.array([0., 0., 1.])

proj_e0 = np.dot(v, e0)
proj_e1 = np.dot(v, e1)
proj_e2 = np.dot(v, e2)



m1 = np.array([[6, -9, 1], [4, 24, 8]])
calc1 = m1 *2 

mi = np.eye(2)
calc2 = mi@m1

m2 = np.array([[4, 3], [3, 2]])
m3 = np.array([[-2, 3], [3, -4]])

calc3 = m2@m3

m4 = np.array([[0,1,2,3,4,5], 
               [10,11,12,13,14,15], 
               [20,21,22,23,24,25], 
               [30,31,32,33,34,35], 
               [40,41,42,43,44,45], 
               [50,51,52,53,54,55]])

s1 = m4[:,1]
s2 = m4[1, 2:4]
s3 = m4[2:4, 4:6]

def swap_rows(M,a,b):
     # Check if M is a numpy array
    if not isinstance(M, np.ndarray):
        raise TypeError("Input M must be a NumPy array")
    
    # Check if M is 2-dimensional
    if M.ndim != 2:
        raise ValueError("Input M must be a 2-dimensional array")
    
    # Check if a and b are integers
    if not (isinstance(a, int) and isinstance(b, int)):
        raise TypeError("Column indices must be integers")
    
    # Check if a and b are valid column indices
    num_cols = M.shape[1]
    if a < 0 or a >= num_cols or b < 0 or b >= num_cols:
        raise ValueError(f"Column indices must be between 0 and {num_cols-1}")
    
    # If cols are the same
    if a == b:
        return M
   
   # Swap columns
    M_copy = M.M_copy()
    M_copy[a], M_copy[b] = M_copy[b], M_copy[a]
    return M_copy


def swap_cols(M, a, b):
       # Check if M is a numpy array
    if not isinstance(M, np.ndarray):
        raise TypeError("Input M must be a NumPy array")
    
    # Check if M is 2-dimensional
    if M.ndim != 2:
        raise ValueError("Input M must be a 2-dimensional array")
    
    # Check if a and b are integers
    if not (isinstance(a, int) and isinstance(b, int)):
        raise TypeError("Column indices must be integers")
    
    # Check if a and b are valid column indices
    num_cols = M.shape[1]
    if a < 0 or a >= num_cols or b < 0 or b >= num_cols:
        raise ValueError(f"Column indices must be between 0 and {num_cols-1}")
    
    # Check if rows are the same
    if a == b:
        return M
    
    # Swap rows
    M_copy = M.copy()
    M_copy[:, [a, b]] = M_copy[:, [b, a]]
    return M_copy


def set_array(L, rows, cols, order='C'):
    # Check if L is a list-type variable
    if not isinstance(L, (list, tuple, np.ndarray)):
        raise TypeError("Input L must be a list, tuple, or numpy array")
    
    # Check if rows and cols are positive integers
    if not (isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0):
        raise ValueError("rows and cols must be positive integers")
    
    # Check if L has the correct number of elements
    if len(L) != rows * cols:
        raise ValueError(f"L must have exactly {rows * cols} elements")
    
    # Check if order is valid
    if order not in ['C', 'F']:
        raise ValueError("order must be either 'C' for row-major or 'F' for column-major")
    
    # Convert L to a numpy array and reshape it
    arr = np.array(L).reshape((rows, cols), order=order)
    
    return arr

L = [1,2,3,4,5,6]
result = set_array(L, 2, 3)




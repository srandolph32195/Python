import numpy as np
import matplotlib.pyplot as plt



def dynamics_solve(f, D = 1, t_0 = 0.0, s_0 = 1, h = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if D == 1:
        S = np.zeros(N + 1)
    
    if D > 1:
        S = np.zeros((N + 1, D))
        
    S[0] = s_0
    
    if method == 'Euler':
        for n in range(N):
            S[n + 1] = S[n] + h * f(T[n], S[n])
    
    if method == 'RK2':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + h/2, S[n] + k1/2)
            S[n + 1] = S[n] + k2
    
    if method == 'RK4':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + h/2, S[n] + k1/2)
            k3 = h * f(T[n] + h/2, S[n] + k2/2)
            k4 = h * f(T[n] + h, S[n] + k3)
            S[n + 1] = S[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            
    return T, np.transpose(S)



def hamiltonian_solve(d_qH, d_pH, D = 1, t_0 = 0.0, q_0 = 0.0, p_0 = 1.0, h = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of Hamiltonian system
    
    - User must specify dimension d of configuration space.
    - Includes Euler, RK2, RK4, Symplectic Euler (SE) and Stormer Verlet (SV) 
      that user can choose from using the keyword "method"
    
    Args:
        d_qH: Partial derivative of the Hamiltonian with respect to coordinates (float for d=1, ndarray for d>1)
        d_pH: Partial derivative of the Hamiltonian with respect to momenta (float for d=1, ndarray for d>1)
        
    Kwargs:
        D: Spatial dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        q_0: Initial position (float for d=1, ndarray for d>1) set to 0.0 as default
        p_0: Initial momentum (float for d=1, ndarray for d>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4", "SE", "SV"
    
    Returns:
        T: Numpy array of times
        Q: Numpy array of positions at the times given in T
        P: Numpy array of momenta at the times given in T
    """
    
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if D == 1:
        P = np.zeros(N + 1)
        Q = np.zeros(N + 1)
    
    if D > 1:
        P = np.zeros((N + 1, D))
        Q = np.zeros((N + 1, D))
        
    Q[0] = q_0
    P[0] = p_0
    
    if method == 'Euler':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n], P[n])
    
    if method == 'RK2':
        for n in range(N):
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * d_qH(Q[n], P[n])
            
            k2_Q = h * d_pH(Q[n] + k1_Q / 2, P[n] + k1_P / 2)
            k2_P = h * d_qH(Q[n] + k1_Q / 2, P[n] + k1_P / 2)
            
            Q[n + 1] = Q[n] + k2_Q
            P[n + 1] = P[n] - k2_P
        
    if method == 'RK4':
        for n in range(N): 
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * d_qH(Q[n], P[n])
            
            k2_Q = h * d_pH(Q[n] + k1_Q / 2, P[n] + k1_P / 2)
            k2_P = h * d_qH(Q[n] + k1_Q / 2, P[n] + k1_P / 2)
            
            k3_Q = h * d_pH(Q[n] + k2_Q / 2, P[n] + k2_P / 2)
            k3_P = h * d_qH(Q[n] + k2_Q / 2, P[n] + k2_P / 2)
            
            k4_Q = h * d_pH(Q[n] + k3_Q, P[n] + k3_P)
            k4_P = h * d_qH(Q[n] + k3_Q, P[n] + k3_P)
            
            Q[n + 1] = Q[n] + (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q) / 6
            P[n + 1] = P[n] - (k1_P + 2 * k2_P + 2 * k3_P + k4_P) / 6
        
    if method == 'SE':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n + 1], P[n])
    
    if method == "SV":
        for n in range(N):
            P_n = P[n] - h / 2 * d_qH(Q[n], P[n])
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P_n)
            P[n + 1] = P_n - h / 2 * d_qH(Q[n + 1], P[n])
        
    return T, np.transpose(Q), np.transpose(P)



def jacobi_rotation(A, i, j):
    """
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # i (int): row parameter.
        # j (int): column parameter.

    #Returns:
        # S (np.ndarray): n by n real symmetric matrix, where the A[j,k] and A[k,j] element is zero
        # J (np.ndarray): n by n orthogonal matrix, the jacobi_rotation matrix
    """

    if A[i][j] != 0:
        T = (A[j][j] - A[i][i]) / (2 * A[i][j])
        if T >= 0:
            t = 1 / (T + np.sqrt(1 + T ** 2))
            
        else:
            t = 1 / (T - np.sqrt(1 + T ** 2))
        
        c = 1 / np.sqrt(1 + t ** 2)
        s = t * c
    
    else:
        c = 1
        s = 0
    
    J = np.identity(np.shape(A)[0])
    J[i][i] = c
    J[i][j] = s
    J[j][i] = -s
    J[j][j] = c
    
    return np.transpose(J) @ A @ J, J    



def off(A):
    """
    Determines the magnitude of the off-diagonal elements of a matrix
    """
    A = np.abs(A) ** 2
    
    return np.sqrt(np.sum(A) - np.sum(np.diag(A)))



def norm(A):
    """
    Determines the magnitude of the elements of a matrix
    """
    A = np.abs(A) ** 2
    
    return np.sqrt(np.sum(A))



def real_eigen(A, tolerance = 1E-5):
    """
    Args:
        A (np.ndarray): n by n real symmetric matrix
    Kwargs:
        tolerance (float): the relative precision (initialized to 1E-5)
    Returns:
        d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
                        to multiplicity and ordered in non-decreasing order
        R (np.ndarray): n by n orthogonal matrix, R[:,i] is the i-th eigenvector
    """
    
    n = np.shape(A)[0]
    R = np.identity(n)
    delta = tolerance * norm(A)
    while off(A) > delta:
        for i in range(n - 1):
            for j in range(i + 1, n):
                A, J = jacobi_rotation(A, i, j)
                R = R @ J
                
    d = np.diag(A) 
    
    return d, R



def hermitian_eigensystem(H, tolerance = 1E-5):
    """
    Args:
        H (np.ndarray): n by n complex hermitian matrix
    Kwargs:
        tolerance (float): the relative precision (initialized to 1E-5)
    Returns:
        d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
                        to multiplicity and ordered in non-decreasing order
        U (np.ndarray): n by n unitary matrix, U[:,i] is the i-th eigenvector
    """
    
    S = np.real(H)
    A = np.imag(H)
    n = np.shape(H)[0]
    O = np.identity(2 * n)
    
    for i in range(n):
        for j in range(n):
            O[i][j] = S[i][j]
            O[i + n][j + n] = S[i][j]
            O[i][j + n] = -A[i][j]
            O[i + n][j] = A[i][j]
    
    d, U_prime = real_eigen(O, tolerance = tolerance)
    U = np.array([[0. + 0.j] * 2 * n] * n)
    
    for i in range(n):
        for j in range(2 * n):
            U[i][j] = U_prime[i][j] + U_prime[i + n][j] * 1j            

    return d, U



def weighted_coin(beta, n, win_earn = 1., lose_earn = -1., plot = True):
    """ Simulates a series of weighted coin flips based on the Metropolis algorithm 
    Args:
        beta (float): gives the probability of winning the coin flip
        n (int): number of flips of the coin
    Kwargs: 
        win_earn (float): amount won on each win
        lose_earn (float): amount lost on each loss
        plot (bool): displays a plot of the running average earned
    Returns:
        avg_earn (float): average earnings per flip
    """
    
    p = [1. - beta, beta]
    S = np.zeros(n)
    avg_S = np.zeros(n)
    earn_tot = np.zeros(n)
    earn = 0
    S[0] = 1
    
    for i in range(1, n):
        S_n = np.random.randint(2)
        
        if S_n == S[i - 1]:
            S[i] = S_n
            
        else:
            p_accept =  min(1., p[S_n] / p[int(S[i - 1])])
            r = np.random.rand(1)
            
            if r < p_accept:
                S[i] = S_n
                
            else:
                S[i] = S[i - 1]
                
        if S[i]:
            earn += win_earn
            
        else:
            earn += lose_earn
        earn_tot[i] = earn / i
 
    avg_earn = np.average(earn_tot)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, n, n), earn_tot)
        ax.set_title('Running Average')
    
    return avg_earn
  


def average_earnings_per_flip(beta, win_earn = 1., lose_earn = -1.):
    """ Computes the theoretical earnings of a weighted coin
    Args:
        beta (float): gives the probability of winning the coin flip
    Kwargs: 
        win_earn (float): amount won on each win
        lose_earn (float): amount lost on each loss
    Returns:
        avg_earn (float): average earnings per flip
    """
    
    avg_earn = beta * win_earn + (1. - beta) * lose_earn

    return avg_earn



def two_dim_ising(L, temp, num_steps, H = 0.):
    """ Initializes an 2D LxL lattice of spin configurations and simulates the evolution of the states using a MCMC algorithm
    Args:
        L (int): Size of one of the sides of the 2D lattice
        temp (float): temperature of the system
        num_steps (int): the number of times the MCMC algorithm is run -- simulates the time the lattice has to evolve
    Kwargs: 
        H (float): strength of the magnetic field
    Returns:
        s (np.array): The 2D LxL spin configurations after the algorithm
        U (np.array): potential energy per site after each iteration
        M (np.array): magnetization per site after each iteration
        X_T (np.array): magnetic susceptibility per site after each iteration
        C_H (np.array): specific heat per site after each iteration
    """
    
    def delta_E(s, i, j):
        s_t = s[(i + 1) % (L-1)][j]
        s_b = s[i - 1][j]
        s_l = s[i][j - 1]
        s_r = s[i][(j + 1) % (L-1)]
        
        return 2 * s[i][j] * (s_t + s_b + s_l + s_r + H)
    
    
    N = L ** 2
    E = 0.
    S = 0.
    E2 = 0.
    S2 = 0.
    U = np.zeros(num_steps + 1)
    M = np.zeros(num_steps + 1)
    X_T = np.zeros(num_steps + 1)
    C_H = np.zeros(num_steps + 1)
    s = [[0] * L] * L
    s = np.array(s)

    for i in range(L):
        for j in range(L):
            s[i][j] = 2 * np.random.randint(2) - 1
            S += s[i][j]
            S2 += s[i][j]**2
            
    for i in range(L):
        for j in range(L):
            s_t = s[int((i + 1) % (L-1))][j]
            s_b = s[i - 1][j]
            s_l = s[i][j - 1]
            s_r = s[i][int((j + 1) % (L-1))]
            E_n = -s[i][j] * (s_t + s_b + s_l + s_r + H)
            E += E_n
            E2 += E_n ** 2
            
    U[0] = E / N
    M[0] = S / N
    X_T[0] = (N - S ** 2) / (N * temp)
    C_H[0] = (E2 - E ** 2) / (N * temp ** 2)
    
    for n in range(num_steps):
        dS = 0
        i = np.random.randint(L)
        j = np.random.randint(L)
        dE = delta_E(s, i, j)
        
        if dE <= 0:
            s[i][j] *= -1
            dS = 2 * s[i][j]
            
        else:
            p_accept = min(1., np.exp(-dE / temp))
            r = np.random.rand(1)
            
            if r < p_accept:
                s[i][j] *= -1
                dS=2 * s[i][j]
            else:
                dE = 0
                
        E += dE
        S += dS
        E2 += dE ** 2
        S2 += dS ** 2
        
        U[n + 1] = U[n] + (E / N - U[n]) / (n + 1)
        M[n + 1] = M[n] + (S / N - M[n]) / (n + 1)
        X_T[n + 1] = X_T[n] + ((S2 - S ** 2) / (N * temp) - X_T[n]) / (n + 1)
        C_H[n + 1] = C_H[n] + ((E2 - E ** 2) / (N * temp ** 2) - C_H[n]) / (n + 1)
                
        
    return s, U, M, X_T, C_H



def weighted_die(weight, n, sides = 1, win_earn = 1., lose_earn = -1.):
    """ Simulates a series of weighted die rolls based on the Metropolis algorithm 
    Args:
        weight (float): gives how much more probable a side is compared to losing probabilities
        n (int): number of rolls of the die
    Kwargs:
        sides (int): how many sides are weighted for winning condition
        win_earn (float): amount won on each win
        lose_earn (float): amount lost on each loss
    Returns:
        avg_earn (float): average earnings per roll
    """
    
    p_l = 1 / (sides * weight + 1)
    p_w = 1 - p_l
    
    avg_earn = weighted_coin(p_w, n, win_earn = win_earn, lose_earn = lose_earn, plot = False)
    
    return avg_earn



def average_earnings_per_roll(weight, sides = 1, win_earn = 1., lose_earn = -1.):
    """ Computes the theoretical earnings of a weighted die
    Args:
        weight (float): gives how much more probable a side is compared to losing probabilities
    Kwargs:
        sides (int): how many sides are weighted for winning condition
        win_earn (float): amount won on each win
        lose_earn (float): amount lost on each loss
    Returns:
        avg_earn (float): average earnings per roll
    """
    
    p_l = 1 / (sides * weight + 1)
    p_w = 1 - p_l
    
    avg_earn = p_w * win_earn + p_l * lose_earn

    return avg_earn



def PID(f, kp, ki, kd, set_val, x0 = 0., n = 1000, dt = 0.000001):
    
    """ Uses PID algorithm to find a set value
    
    Args:
        f: A python function which is used to represent a system
        kp (float): proportionality constant
        ki (float): integral constant
        kd (float): derivative constant
        set_val (float): the value for which the system should tend towards
        
    Kwargs:
       x0 (float): the value which the algorithm starts at
       n (int): number of iterations the algorithm runs for
       dt (float): time step between each iteration -- used in integral and derivative terms
    
    Returns:
        x_vals (array): the x values returned by the algorithm
        y_vals (array): the y values returned by the algorithm
    """
        
    error_integral = 0
    x = x0
    x_vals = np.zeros(n)
    y_vals = np.zeros(n)
    
    current_val = f(x)   
    x_vals[0] = x        
    y_vals[0] = current_val
    previous_error = set_val - current_val


    for i in range(n - 1):
        error = set_val - current_val
        error_integral += error * dt
        error_derivative = (error - previous_error) / dt
        adjustment = kp * error + ki * error_integral + kd * error_derivative
        
        x += adjustment    
        current_val = f(x)
        previous_error = error
        x_vals[i+1] = x
        y_vals[i+1] = current_val

        
    return x_vals, y_vals

        
        
def PID_extrema(f, kp, ki, kd, extrema, x0 = 0., n = 1000, dt = 0.000001):
    
    """ Uses PID algorithm and the change in the function to find a nearby extrema
    
    Args:
        f: A python function which is used to represent a system
        kp (float): proportionality constant
        ki (float): integral constant
        kd (float): derivative constant
        extrema ('max', 'min'): the type of extrema the algorithm should tend towards
        
    Kwargs:
       x0 (float): the value which the algorithm starts at -- must be on the side of the opposite extrema nearest to desired extrema (i.e.
           x0 needs to be to the left of a min if said min is to the right of the desired max, or to the right of a min if said min is to
           the left of the desired max)
       n (int): number of iterations the algorithm runs for
       dt (float): time step between each iteration
    
    Returns:
        x_vals (array): the x values returned by the algorithm
        y_vals (array): the y values returned by the algorithm
    """
        
    error_integral = 0
    x = x0
    x_vals = np.zeros(n)
    y_vals = np.zeros(n)
    current_val = (f(x + dt) - f(x)) / dt
    x_vals[0] = x        
    y_vals[0] = f(x)
    previous_error = current_val


    for i in range(n - 1):
        error = current_val
        error_integral += error * dt
        error_derivative = (error - previous_error)/dt
        adjustment = kp * error + ki * error_integral + kd * error_derivative
        
        if extrema == 'max':
            x += adjustment 
        elif extrema == 'min':
            x -= adjustment
            
        current_val = (f(x + dt) - f(x)) / dt
        previous_error = error
        x_vals[i + 1] = x
        y_vals[i + 1] = f(x)
        

    return x_vals, y_vals
    
 

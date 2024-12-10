import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Analytic integration exceeded time limit.")


def simpson_coefficients(left, right, N):
    if N % 2 == 0:
        raise ValueError("N must be odd for Simpson's rule")
    
    # Calculate step size
    h = (right - left) / (N - 1)
    
    # Generate x values
    x = np.linspace(left, right, N)
    
    # Create coefficient array
    coef = np.full(N, 2 * h / 3)  # Default to 2h/3 for all entries
    coef[1:N-1:2] = 4 * h / 3     # Set odd indices to 4h/3
    coef[0] = coef[-1] = h / 3    # Set first and last to h/3
    
    return x, coef
def cumulative_integral(f_x,left,right,N=2049):
    """
    Compute cumulative integral values over the range [x[0], x[-1]].

    Parameters:
        x (array): The points in the range.
        y (array): Function values at points x.
        coefficients (array): Integration coefficients (e.g., from Simpson or Trapezoidal).

    Returns:
        array: Cumulative integral values at each x.
    """
    x,w=simpson_coefficients(left,right,N)
    cumulative = np.cumsum(w * f_x(x))
    def get_integral(x):
        # scale x to [0;1]
        x-=left
        if right!=left:
            x = x/(right-left)
        
        # get relative pos
        index=np.int32(x*N)
        
        bigger = index>=len(cumulative)
        is32 = isinstance(index,np.int32)
        if np.any(bigger):
            if is32:
                index=len(cumulative)-1    
            else:
                index[bigger]=len(cumulative)-1
        
        smaller = index<0
        if np.any(smaller):
            if is32:
                index=0
            else:
                index[smaller]=0
        
        return cumulative[index]
    return get_integral,x,cumulative

def cached_approx_integral(f):
    saved_integrals = {}
    def get_integral(x,a):
        x_min = np.min(x)
        x_max = np.max(x)
        key = (x_min,x_max,a)
        if key not in saved_integrals.keys():
            saved_integrals[key] = cumulative_integral(lambda t: f(t,a),x_min,x_max)[0]
        return saved_integrals[key](x)
    return get_integral

class InhomogeneousLinearDifferentialEquation:
    """Tries to build solution to system of linear differential equations with inhomogeneous part"""
    def __init__(self,A,f_t,t0,x0,integral_timeout_sec=2):
        self.A=A
        self.x0=x0

        # some complex stuff to get required integral of e^(at)*f_i(t) for each f_i
        t,a = sp.var("t a")
        f_values = f_t(t)
        
        
        # Set timeout for the integration process
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # symbolic integrals
        integral_sp = []
        # their numerical counterpart
        integral_num = []
        for f in f_values:
            integral_expr=sp.exp(a*t)*f
            try:
                # Set the timeout alarm
                signal.alarm(integral_timeout_sec)
                analytic = sp.integrate(integral_expr,t)
                integral_sp.append(analytic)
                analytic=lambdify("t,a",analytic)
                integral_num.append(analytic)
                signal.alarm(0)
            except TimeoutException:
                print(f"Timeout occurred during analytic integration of {f}. Using approximate integration.")
                signal.alarm(0)
                integral_expr = lambdify("t,a", integral_expr)
                integral_num.append(cached_approx_integral(integral_expr))
            except Exception as e:
                print(f"Failed to build analytic integral of {f}")
                print(e)
                print("will use approximate integration")
                integral_expr=lambdify("t,a",integral_expr)
                integral_num.append(cached_approx_integral(integral_expr))
            signal.alarm(0)

        self.integral=lambda t,a: np.stack([i(t,a) for i in integral_num])

        # get numpy-version of symbolic f_t
        v_t = lambdify("t",f_t(t))
        self.f_t = lambda x: v_t(x).flatten()

        # compute eigendecomposition of matrix
        D,T = np.linalg.eig(A)
        self.T=T
        self.D=D
        self.T_inv = np.linalg.inv(T)

        # compute constants to match X(0)=x0
        self.constants = self.compute_constant(t0,x0)

    # for each f_i_t compute integral e^(at) f_i_t(t) dt    
    def F_t(self,t,a):
        integ = self.integral(t,a)
        res = np.array([
            integ[i] for i in range(len(integ))
        ])
        return res

    def S_i(self,t,i):
        return self.F_t(t,-self.D[i])

    def S(self,t):
        # return np.array([self.S_i(t,i) for i in range(len(self.x0))])[:,:,0]
        s= np.array([self.S_i(t,i) for i in range(len(self.x0))])
        return s

    def analytic_solution(self,t):
        """Get analytic solution of a inhomogeneous system of differential equations"""
        t=np.array(t)
        S = np.array([self.S_i(t,i) for i in range(len(self.x0))]).swapaxes(2,0).swapaxes(1,2)
        exp_part = np.exp(self.D[np.newaxis,:]*t[:,np.newaxis])
        v1 = self.T * exp_part[:,None,:]
        v2 = self.T_inv * S @ np.ones_like(self.constants)+self.constants
        return np.einsum('ijk,ik->ij', v1, v2)

    def compute_constant(self,t0,x0):
        return  self.T_inv * np.exp(-self.D*t0) @ x0-self.T_inv * self.S(t0) @ np.ones_like(x0)
    
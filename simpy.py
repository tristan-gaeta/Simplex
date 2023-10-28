__package__ = 'simpy'
__author__ = 'Tristan Gaeta'
__doc__ = """The SimPy library can solve linear programs using the simplex method

SimPy uses the two phase method to solve LPs. All problems are converted
to canonical form and use a phase-1 LP to find an initial basic feasible 
solution.

See the SimPy user guide for more details.
"""

import numpy as np


def __check_dims__(A: np.matrix, b: np.ndarray, c: np.ndarray) -> None:
    """Checks dimensions for given LP

    This function will raise an exception when the dimensions of the
    given LP are incompatible.
    """

    # check for proper dimensions
    if A.ndim != 2:
        raise ValueError(f'Argument A must be 2 dimensional.')

    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(
            f'Arguments A and b have incompatible dimensions: {A.shape} and {b.shape}. Should be (m,n) and (m,).')
    if c.shape != (n,):
        raise ValueError(
            f'Arguments A and c have incompatible dimensions: {A.shape} and {c.shape}. Should be (m,n) and (n,).')


def __canonize__(c, A_eq=None, b_eq=None, A_leq=None, b_leq=None, A_geq=None, b_geq=None, unbounded=False) -> tuple[np.matrix, np.ndarray, np.ndarray]:
    """Convert the given LP to canonical form

    This function also serves to cast input ArrayLike objects to float ndarrays.
    """

    # convert to float ndarrays
    c = np.asfarray(c)
    x_dim = c.shape[0]

    A_eq = np.empty((0, x_dim)) if A_eq is None else np.asfarray(A_eq)
    A_leq = np.empty((0, x_dim)) if A_leq is None else np.asfarray(A_leq)
    A_geq = np.empty((0, x_dim)) if A_geq is None else np.asfarray(A_geq)

    b_eq = np.empty(0) if b_eq is None else np.asfarray(b_eq)
    b_leq = np.empty(0) if b_leq is None else np.asfarray(b_leq)
    b_geq = np.empty(0) if b_geq is None else np.asfarray(b_geq)

    # Check for proper dimensions
    __check_dims__(A_eq, b_eq, c)
    __check_dims__(A_leq, b_leq, c)
    __check_dims__(A_geq, b_geq, c)

    # we will treat inequalities the same by multiplying geq constraints by -1
    A_ineq = np.vstack((A_leq, -A_geq))

    num_ineq, _ = A_ineq.shape
    num_eq, _ = A_eq.shape

    # --Canonical Form--
    b = np.hstack((b_eq, b_leq, -b_geq))
    A = np.block([
        [A_eq, np.zeros((num_eq, num_ineq))],
        [A_ineq, np.eye(num_ineq)]
    ])
    c = np.hstack((c, np.zeros(num_ineq)))

    if unbounded:
        A = np.hstack(A, -A)
        c = np.hstack(c, -c)

    return (A, b, c)


def __init_problem__(A: np.matrix, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find an initial basic feasible solution to the LP in canonical form"""

    m, n = A.shape

    # set b >= 0 so that initial soln is feasible (x>=0 satisfied)
    neg_vals = b < 0
    A[neg_vals] *= -1
    b[neg_vals] *= -1

    # Find initial feasible solution
    if np.array_equal(A[:, n-m:], np.eye(m)):   # A has identity at the end
        # origin is feasible
        x = np.hstack((np.zeros(n-m), b))  # x = (0,b)
        basic_vars = np.array(range(n-m, n))
        nonbasic_vars = np.array(range(n-m))
    else:  # We need to find an initial soln
        A1 = np.hstack((A, np.eye(m)))
        c1 = np.hstack((np.zeros(n), -np.ones(m)))

        # Solve Phase 1 LP
        optimum, x1 = simplex(A1, b, c1)

        if optimum != 0:
            raise ValueError('This problem is infeasible.')

        # our initial soln
        x = x1[:n]

        # it is possible for some nonbasic vars to be zero, so
        # well just say the basic vars are the m largest vars
        vars = np.argsort(x)
        basic_vars = vars[n-m:]
        nonbasic_vars = vars[:n-m]
    return (x, basic_vars, nonbasic_vars)


def simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray, form: str = 'canonical', minimize: bool = False, print_steps: bool = False, **kwargs) -> tuple[float, np.ndarray]:
    """Perform the simplex algorithm on the given LP with positive variable constraints

    Parameters
    ----------
    A : matrix
        The constraint matrix of the LP (a 2D ndarray).
    b : ndarray
        The constraint vector (a 1D ndarray).
    c : ndarray
        The objective value.
    form : str, optional
        Either 'canonical', 'standard', or 'normal' (Default is 'canonical').
    minimize : bool, optional
        Will solve for minimum if set to true (Default is False).
    print_state : bool, optional
        Each iteration will be printed if true (Default is False).
    kwargs : dict[str, Any], optional
        Additional keyword arguments. May include A_eq and b_eq, A_leq and b_leq, and A_geq
        and b_geq for =, ≤, and ≥ constraints respectively

    Returns
    -------
    cTx : float
        The optimal value of the objective function
    x : ndarray
        The optimal solution to the LP
    """

    initial_size = len(c)

    # convert to canonical form
    match (form):
        case 'canonical' | 'eq':
            A, b, c = __canonize__(c, A_eq=A, b_eq=b,  **kwargs)
        case 'standard' | 'leq':
            A, b, c = __canonize__(c, A_leq=A, b_leq=b,  **kwargs)
        case 'normal' | 'geq':
            A, b, c = __canonize__(c, A_geq=A, b_geq=b,  **kwargs)
        case _:
            raise ValueError(
                'Valid options for form are "canonical", "standard", "normal" for =, ≤, and ≥ constraints respectively.')

    # simply maximize -c
    if minimize:
        c *= -1

    # find basic feasible solution
    x, basic_vars, nonbasic_vars = __init_problem__(A, b)

    # begin algo
    while True:
        # update variables
        B = A[:, basic_vars]
        N = A[:, nonbasic_vars]
        cB = c[basic_vars]
        cN = c[nonbasic_vars]
        xB = x[basic_vars]

        # Step 1: find reduced costs
        y = np.linalg.solve(B.T, cB)
        reduced_costs = cN - N.T @ y

        if print_steps:
            print(f'x = {x}')
            print(f'Basic Vars: {basic_vars}')
            print(f'Non-Basic Vars: {nonbasic_vars}')
            print(f'Reduced Cost: {reduced_costs}\n')

        # Step 2: check for optimality
        if np.max(reduced_costs) <= 0:  # if all non-positive
            # truncate slack/surplus vars that we introduced
            c = c[:initial_size]
            x = x[:initial_size]
            if minimize:
                # undo negation of c
                c *= -1
            return (c@x, x)

        # Step 3: Choose entering var and compute simplex direction
        # this will return the lowest index that is positive
        entering = np.argmax(reduced_costs > 0)
        dB = np.linalg.solve(B, -A[:, nonbasic_vars[entering]])

        # Step 4: find maximum step size
        if np.min(dB) >= 0:
            raise ValueError('This problem is unbounded.')

        with np.errstate(divide='ignore'):  # ignore divide by zero warning
            ratios = xB / -dB

        # ignore components with nonnegative values by setting ratio to infinity
        ratios[dB >= 0] = float('inf')

        leaving = np.argmin(ratios)
        step_size = ratios[leaving]

        # Step 5: Update variable
        x[basic_vars] += step_size * dB
        x[nonbasic_vars[entering]] = step_size
        x[basic_vars[leaving]] = 0

        # swap entering and leaving variables
        nonbasic_vars[entering], basic_vars[leaving] = basic_vars[leaving], nonbasic_vars[entering]
        # sorting will prevent infinite loops in the algo
        nonbasic_vars.sort()
        basic_vars.sort()

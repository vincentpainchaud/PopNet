"""Dynamical systems related to the Wilson--Cowan model.

This modules implements various dynamical systems related to the Wilson--Cowan
model. It first implements an abstract base class `DynamicalSystem`, from which
several other classes are derived to represent different dynamical systems.
These are listed in the [Classes And Hierarchy](#classes-and-hierarchy)
section below.

Classes and hierarchy
---------------------

The important classes of the module are summarized below. The indentation
follows the hierarchy.

 - `DynamicalSystem` : An abstract base class to represent a dynamical system.
     - `WilsonCowanSystem` : An equivalent to the classical Wilson--Cowan
       dynamical system.
     - `MeanFieldSystem` : The Wilson--Cowan system with refractory state.
     - `MixedSystem` : An extension of the last case where the refractory state
       is weighted.
     - `TaylorExtendedSystem` : An extended Wilson--Cowan system with refractory
       state and covariances, obtained from a moment closure based on a
       second-order Taylor approximation.
     - `ExtendedSystem` : An extended Wilson--Cowan system with refractory state
       and covariances, obtained from a moment closure based on the shape of
       sigmoid functions.

"""

import numpy as np
from scipy.optimize import root
from warnings import warn

from .exceptions import *
from . import _internals
from . import structures
from . import graphics


class DynamicalSystem:
    """Represent dynamical systems.

    `DynamicalSystem` is an abstract base class intended to represent dynamical
    systems in PopNet. Each subclass must implement a vector field, and the
    base class has several methods to study this vector field. For example, a
    method is available to find equilibrium points. A subclass can also
    implement a Jacobian matrix, in which case methods are available to find
    its eigenvalues and eigenvectors.

    !!! note
        PopNet assumes that any subclass of `DynamicalSystem` implements the
        method `DynamicalSystem.vector_field` and sets the property
        `DynamicalSystem.dim` at initialization.

    Parameters
    ----------
    config : popnet.structures.Configuration
        A configuration associated with the dynamical system.

    Attributes
    ----------
    config : popnet.structures.Configuration
        Configuration associated with the dynamical system. See
        `DynamicalSystem.config`.
    dim : int
        Dimension of the dynamical system. See `DynamicalSystem.dim`.

    """

    def __init__(self, config):
        self.config = config
        self._dim = self._get_dimension()

    @property
    def config(self):
        """Configuration used with the dynamical system.

        Configuration defining all parameters used in the dynamical system. It
        must be a `popnet.structures.Configuration` instance. It cannot be
        deleted.
        """
        return self._config

    @config.setter
    def config(self, new_value):
        if not isinstance(new_value, structures.Configuration):
            raise TypeError('The configuration used with a dynamical system '
                            'must be a \'Configuration\' instance.')
        self._config = new_value

    @property
    def dim(self):
        """Dimension of the dynamical system.

        Dimension of the dynamical system. It is set at initialization, and it
        cannot be reset nor deleted afterwards.
        """
        return self._dim

    def find_equilibrium_near(self, state, verbose=True, method='hybr'):
        """Find an equilibrium point near at a given state.

        Find an equilibrium point of the dynamical system near the given state.
        This method uses the [`root`](https://31c8.short.gy/scipy-optimize-root)
        function from SciPy's `optimize` module.

        Parameters
        ----------
        state : array_like
            The initial guess for the equilibrium point. If it has more
            components than the dynamical system, only the first components are
            used.
        verbose : bool, optional
            If `True`, a warning is issued if the optimizer fails and no
            equilibrium point is found. Defaults to `True`.
        method : str, optional
            The solver used to find the find fixed point. It must be one of
            the accepted values for the corresponding argument of `root`.
            Defaults to `'hybr'`.

        Returns
        -------
        array_like
            The equilibrium point found, or `None` if the optimization failed.

        Warns
        -----
        popnet.exceptions.PopNetWarning
            If `verbose` is `True` and the optimizer did not succeed.
        """
        state = np.array(state, float)
        try:
            self.jac(state)
        except NotImplementedError:
            jac = None
        else:
            jac = self.jac
        sol = root(self.vector_field, state[:self.dim], jac=jac)
        if not sol.success:
            if verbose:
                warn('The optimizer did not succeed.',
                     category=PopNetWarning, stacklevel=2)
            return None
        return sol.x

    def get_eigs_at(self, state):
        """Get eigenvalues and eigenvectors of the Jacobian matrix.

        Get the eigenvalues and eigenvectors of the Jacobian matrix
        corresponding to the linearization of the dynamical system, evaluated
        at the given state. If eigenvectors are not needed,
        `DynamicalSystem.get_eigenvals_at` should be used instead.

        Parameters
        ----------
        state : array_like
            The state at which the Jacobian matrix is to be evaluated.

        Returns
        -------
        array
            The eigenvalues, repeated according to their multiplicities, and
            sorted from largest to lowest real part.
        array
            The associated eigenvectors.

        Raises
        ------
        NotImplementedError
            If the Jacobian matrix for this system is not implemented.
        numpy.LinAlgError
            If the eigenvalue computation does not converge.
        """
        eigenvals, eigenvects = np.linalg.eig(self.jac(state))
        argsort = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[argsort]
        eigenvects = eigenvects[argsort]
        return eigenvals, eigenvects

    def get_eigenvals_at(self, state):
        """Get eigenvalues of the Jacobian matrix at a given state.

        Get the eigenvalues of the Jacobian matrix corresponding to the
        linearization of the dynamical system, evaluated at the given state.

        Parameters
        ----------
        state : array_like
            The state at which the Jacobian matrix is to be evaluated.

        Returns
        -------
        array
            The eigenvalues, repeated according to their multiplicities, and
            sorted from largest to lowest real part.

        Raises
        ------
        NotImplementedError
            If the Jacobian matrix for this system is not implemented.
        numpy.LinAlgError
            If the eigenvalue computation does not converge.
        """
        eigenvals = np.linalg.eigvals(self.jac(state))
        eigenvals = np.sort(eigenvals)[::-1]
        return eigenvals

    def get_phase_plane(self, axes, fixed_axes=0., values=None, name=None):
        """Get a phase plane.

        Get a phase plane with given independant variables to draw it later. For
        this method to be available, the dynamical system must has at least two
        dimensions.

        Parameters
        ----------
        axes : tuple of int
            Axes indicating the independant variables that will be the phase
            plane's axes.
        fixed_axes : array_like or float, optional
            Determines the values of the remaining axes other than those chosen
            with `axes`. If it is a float, all other axes are set to this
            value. If it is an array, its length must be the dimension of the
            dynamical system minus two, and in that case every axis is fixed at
            the value given by `fixed_axes`, in the order of the system,
            skipping the axes chosen by `axes`. It is ignored is the dynamical
            system is two-dimensional. Defaults to 0.
        name : str, optional
            A name associated with the phase plane. Defaults to `None`, in which
            case it is replaced with `'Phase plane'`.

        Returns
        -------
        popnet.graphics.PhasePlane
            A phase plane to be drawn.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the dynamical system has only one dimension.
        """
        if (n := self.dim - 2) < 0:
            raise PopNetError('Can\'t draw a phase plane for a dynamical system'
                              ' of only one dimension.')
        if n == 0:
            fix = None
        else:
            fix = fixed_axes
        return graphics.PhasePlane(self, axes=axes, fixed_axes=fix, name=name)

    def jac(self, state):
        """Get the Jacobian matrix evaluated at a given state.

        Parameters
        ----------
        state : array_like
            The state at which the Jacobian matrix is to be evaluated.

        Returns
        -------
        array
            The Jacobian matrix evaluated at the state.

        Raises
        ------
        NotImplementedError
            If the Jacobian matrix for this system is not implemented.
        """
        raise NotImplementedError('No Jacobian matrix implemented for this '
                                  'system.')

    def vector_field(self, state):
        """Get the vector field evaluated at a given state.

        Get the vector field corresponding to the dynamical system evaluated
        at a given state.

        Parameters
        ----------
        state : array_like
            The state at which the vector field is to be evaluated.

        Returns
        -------
        array
            The vector field evaluated at the state.
        """
        raise NotImplementedError('A dynamical system must implement a vector '
                                  'field.')

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        raise NotImplementedError('A dynamical system must give its dimension.')


class WilsonCowanSystem(DynamicalSystem):
    """Dynamical system for the Wilson--Cowan model.

    Specializes `DynamicalSystem` for an equivalent to the original
    Wilson--Cowan model, without refractory state or correlations. For *p*
    populations, a state of this system has the form
    \\[
        (A_1, ..., A_p),
    \\]
    where \\(A_J\\) is the expectation of the activity of the *J*th population,
    in the order given by the list of populations in the configuration's
    network.

    The Jacobian matrix is implemented for this system.

    """

    def jac(self, state):
        """Jacobian matrix of the vector field. 

        See `DynamicalSystem.jac` for details.
        """
        p = len(self.config.network.populations)
        A = state
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        j = np.zeros((p, p))
        for J, popJ in enumerate(self.config.network.populations):
            rJ = 1 + popJ.beta / popJ.gamma
            SJ = 1 - rJ * A[J]
            j[J,J] = (- popJ.beta - popJ.alpha * rJ * popJ.F(B[J])
                        + popJ.alpha * SJ * popJ.dF(B[J])
                            * self.config.network.c[J,J])
            for K, popK in enumerate(self.config.network.populations):
                if K != J:
                    j[J,K] = (popJ.alpha * SJ * popJ.dF(B[J])
                                * self.config.network.c[J,K])
        return np.array(j, float)

    def vector_field(self, state):
        """Vector field of the Wilson--Cowan model. 

        See `DynamicalSystem.vector_field` for details.
        """
        p = len(self.config.network.populations)
        A = state
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        f = np.zeros(p)
        for J, popJ in enumerate(self.config.network.populations):
            SJ = 1 - (1 + popJ.beta / popJ.gamma) * A[J]
            f[J] = - popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * SJ
        return np.array(f, float)

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        return len(self.config.network.populations)


class MeanFieldSystem(DynamicalSystem):
    """Dynamical system for the Wilson--Cowan model with refractory state.

    Specializes `DynamicalSystem` for the Wilson--Cowan model with refractory
    state explicitely included. Covariances are not considered in this case.
    For *p* populations, a state of this system has the form
    \\[
        (A_1, ..., A_p, R_1, ..., R_p),
    \\]
    where \\(A_J\\) and \\(R_J\\) are respectively the expectations of the
    active and refractory fractions of the *J*th population, in the order given
    by the list of populations in the configuration's network.

    The Jacobian matrix is implemented for this system.

    """

    def jac(self, state):
        """Jacobian matrix of the vector field. 

        See `DynamicalSystem.jac` for details.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p : 2*p]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        j = np.zeros((2*p, 2*p))
        for J, popJ in enumerate(self.config.network.populations):
            j[J,J] = (- popJ.beta - popJ.alpha*popJ.F(B[J]) + popJ.alpha
                        * popJ.dF(B[J])*self.config.network.c[J,J]*S[J] )
            j[J,J+p] = - popJ.alpha*popJ.F(B[J])
            j[J+p,J] = popJ.beta
            j[J+p,J+p] = - popJ.gamma
            for K, popK in enumerate(self.config.network.populations):
                if K != J:
                    j[J,K] = (popJ.alpha*popJ.dF(B[J]) 
                                * self.config.network.c[J,K] * S[J])
        return np.array(j, float)

    def vector_field(self, state):
        """Vector field of the Wilson--Cowan model with refractory state. 

        See `DynamicalSystem.vector_field` for details.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p :]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        f = np.zeros(2*p)
        for J, popJ in enumerate(self.config.network.populations):
            f[J] = - popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * S[J]
            f[J+p] = - popJ.gamma * R[J] + popJ.beta * A[J]
        return np.array(f, float)

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        return 2 * len(self.config.network.populations)


class MixedSystem(DynamicalSystem):
    """Dynamical system for the 'mixed' Wilson--Cowan model.

    Specializes `DynamicalSystem` to study the transition between the classical
    Wilson--Cowan model and its extension with refractory state. This class can
    be seen as a combination of the `WilsonCowanSystem` and `MeanFieldSystem`
    classes. Covariances are not considered in this case. For *p* populations,
    a state of this system has the form
    \\[
        (A_1, ..., A_p, R_1, ..., R_p),
    \\]
    where \\(A_J\\) and \\(R_J\\) are respectively the expectations of the
    active and refractory fractions of the *J*th population, in the order given
    by the list of populations in the configuration's network.

    The Jacobian matrix is implemented for this system.

    In this case the system has an additional data attribute `epsilon`, which
    is a `float` and has a value between 0 and 1. It defines how much the
    refractory state is considered. See `MixedSystem.epsilon` for details.

    """

    def __init__(self, config, epsilon=1):
        self.epsilon = epsilon
        super().__init__(config)

    @property
    def epsilon(self):
        """Transition parameter for the refractory state.

        Float parameter with value between 0 and 1 that determines 'how much'
        the refractory state is considered.

        - When `epsilon` is 1 the refractory state is fully considered, and the
        vector field is the same as that of the `MeanFieldSystem` class.
        - When `epsilon` has a value between 0 and 1, the derivative of the
        refractory state's components is the same as in the `MeanFieldSystem`
        class, but multiplied by `1/epsilon`, so these components converge
        towards their equilibrium solutions faster than they would normally.
        - The case where `epsilon` is zero would be the case where the
        refractory state is set to its equilibrium solution and the vector
        field is that of Wilson--Cowan's model. However, for this case, the
        `WilsonCowanSystem` class must be used instead to avoid divisions by
        zero.

        This property cannot be deleted.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_value):
        try:
            new_value = float(new_value)
        except Exception:
            raise TypeError('\'epsilon\' must be a float.')
        if not 0 < new_value <= 1:
            if new_value == 0:
                msg = ('\'epsilon\' can\'t be equal to 0. For this case, use'
                       ' the \'WilsonCowanSystem\' class instead.')
            else:
                msg = '\'epsilon\' has to be between 0 and 1.'
            raise ValueError(msg)
        self._epsilon = new_value

    def find_bifurcation_near(self, state, precision):
        """Find a bifurcation point near a given state.

        Find a value of epsilon where the dynamical system undergoes a
        bifurcation near a given state. If there is more than one such
        bifurcation point, only the first value of epsilon is found.

        Parameters
        ----------
        state : array_like
            State near which to find a bifurcation. To find the precise steady
            state, `MixedSystem.find_equilibrium_near` is called with `state`.
        precision : int
            Number of significant digits on the value of epsilon at the
            bifurcation point.

        Returns
        -------
        float
            Value of epsilon at which the bifurcation occurs, or `None` if no
            bifurcation is found.
        array_like
            Eigenvalues of the Jacobian matrix at the bifurcation point, or
            `None` is no bifurcation is found.

        Warns
        -----
        PopNetWarning
            If no bifurcation is found.
        """
        finder = _BifurcationFinder(self, state)
        e, eigenvals = finder.run(precision)
        if e is None:
            warn('No bifurcation found.', category=PopNetWarning, stacklevel=2)
        return e, eigenvals

    def jac(self, state):
        """Jacobian matrix of the vector field.

        See `DynamicalSystem.jac` for details.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p : 2*p]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        j = np.zeros((2*p, 2*p))
        for J, popJ in enumerate(self.config.network.populations):
            j[J,J] = (- popJ.beta - popJ.alpha*popJ.F(B[J]) + popJ.alpha
                        * popJ.dF(B[J])*self.config.network.c[J,J]*S[J] )
            j[J,J+p] = - popJ.alpha*popJ.F(B[J])
            j[J+p,J] = popJ.beta / self.epsilon
            j[J+p,J+p] = - popJ.gamma / self.epsilon
            for K, popK in enumerate(self.config.network.populations):
                if K != J:
                    j[J,K] = (popJ.alpha*popJ.dF(B[J]) 
                                * self.config.network.c[J,K] * S[J])
        return np.array(j, float)

    def vector_field(self, state):
        """Vector field of the 'mixed' Wilson--Cowan model. 

        See `DynamicalSystem.vector_field`.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p :]
        S = 1 - A - R
        B = self.config.Q.copy()
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
        f = np.zeros(2*p)
        for J, popJ in enumerate(self.config.network.populations):
            f[J] = - popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * S[J]
            f[J+p] = 1/self.epsilon * (- popJ.gamma * R[J] + popJ.beta * A[J])
        return np.array(f, float)

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        return 2 * len(self.config.network.populations)


class TaylorExtendedSystem(DynamicalSystem):
    """Dynamical system for the extended Wilson--Cowan model.

    Specializes `DynamicalSystem` for the extended Wilson--Cowan model obtained
    from the closure that uses a second-order Taylor approximation. Here the
    refractory state and the covariances between fractions of populations are
    included. For *p* populations, a state of this system has the form
    \\[
        (\\begin{aligned}[t]
            & A_1, A_2, ..., A_p, \\\\
            & R_1, R_2, ..., R_p, \\\\
            & \\mathrm{C}_{AA}^{11}, \\mathrm{C}_{AA}^{12}, ..., 
                \\mathrm{C}_{AA}^{1p}, \\mathrm{C}_{AA}^{22}, ..., 
                \\mathrm{C}_{AA}^{2p}, \\mathrm{C}_{AA}^{33}, ..., 
                \\mathrm{C}_{AA}^{3p}, ..., \\mathrm{C}_{AA}^{pp}, \\\\
            & \\mathrm{C}_{RR}^{11}, \\mathrm{C}_{RR}^{12}, ..., 
                \\mathrm{C}_{RR}^{1p}, \\mathrm{C}_{RR}^{22}, ..., 
                \\mathrm{C}_{RR}^{2p}, \\mathrm{C}_{RR}^{33}, ..., 
                \\mathrm{C}_{RR}^{3p}, ..., \\mathrm{C}_{RR}^{pp}, \\\\
            & \\mathrm{C}_{AR}^{11}, \\mathrm{C}_{AR}^{12}, ..., 
                \\mathrm{C}_{AR}^{1p}, \\mathrm{C}_{AR}^{21}, ...,
                \\mathrm{C}_{AR}^{2p}, ..., \\mathrm{C}_{AR}^{p1}, 
                \\mathrm{C}_{AR}^{p2}, ..., \\mathrm{C}_{AR}^{pp})
        \\end{aligned}
    \\]
    where \\(A_J\\) and \\(R_J\\) are respectively the expectations of the
    active and refractory fractions of the *J*th population, in the order given
    by the list of populations in the configuration's network, and
    \\(\\mathrm{C}_{XY}^{JK}\\) is the covariance between the fractions
    \\(X_J\\) and \\(Y_K\\), where \\(X\\) and \\(Y\\) stand for random
    variables associated with active or refractory fractions.

    In this system, the Jacobian matrix is implemented only for the case where
    the network has only one population.

    Notes
    -----
    The case where the network has only one population is actually handled in a
    separate (private) class `_TaylorExtendedSystemOne`, which uses a simpler
    implementation of the vector field and implements the Jacobian matrix. The
    class constructor of `TaylorExtendedSystem` automatically instantiates
    `_TaylorExtendedSystemOne` when the network has only one population.

    This is considered to be an implementation detail. Only the class
    `TaylorExtendedSystem` should be called by a user.

    """

    def __new__(cls, config, **kwargs):
        if not isinstance(config, structures.Configuration):
            raise TypeError('The configuration used with a \'DynamicalSystem\' '
                            'instance must be a \'Configuration\' instance.')
        if len(config.network.populations) == 1:
            return super().__new__(_TaylorExtendedSystemOne)
        return super().__new__(cls)

    def vector_field(self, state):
        """Vector field of the extended Wilson--Cowan model.

        See `DynamicalSystem.vector_field` for details.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p : 2*p]
        S = 1 - A - R
        CAA = _internals._unflat_scalar_triangle(
                state[2*p : 2*p + round(p*(p+1)/2)])
        CRR = _internals._unflat_scalar_triangle(
                state[2*p + round(p*(p+1)/2) : 2*p+p*(p+1)])
        CAR = (state[2*p + p*(p+1) :]).reshape((p,p))
        CAS = - CAA - CAR
        CSR = - CRR - CAR
        B = self.config.Q.copy()
        VarB = np.zeros(p)
        CAB = np.zeros((p,p))
        CRB = np.zeros((p,p))
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
            for L in range(p):
                VarB[J] += (self.config.network.c[J,K] 
                            * self.config.network.c[J,L] * CAA[K,L])
                CAB[J,K] += self.config.network.c[K,L] * CAA[J,L]
                CRB[J,K] += self.config.network.c[K,L] * CAR[L,J]
        f = np.zeros(self.dim)
        dCAA = np.zeros((p,p))
        dCRR = np.zeros((p,p))
        dCAR = np.zeros((p,p))
        for J, popJ in enumerate(self.config.network.populations):
            f[J] = (- popJ.beta * A[J] + popJ.alpha*popJ.F(B[J]) * S[J]
                    - popJ.alpha*popJ.dF(B[J]) * (CAB[J,J] + CRB[J,J])
                    + popJ.alpha/2*popJ.ddF(B[J]) * S[J] * VarB[J])
            f[J+p] = - popJ.gamma * R[J] + popJ.beta * A[J]
            for K, popK in enumerate(self.config.network.populations):
                dCAA[J,K] = (- (popJ.beta + popK.beta) * CAA[J,K]
                                + popJ.alpha*popJ.F(B[J]) * CAS[K,J]
                                + popK.alpha*popK.F(B[K]) * CAS[J,K]
                                + popJ.alpha*popJ.dF(B[J]) * S[J] * CAB[K,J]
                                + popK.alpha*popK.dF(B[K]) * S[K] * CAB[J,K])
                dCRR[J,K] = (- (popJ.gamma + popK.gamma) * CRR[J,K]
                                + popJ.beta * CAR[J,K] + popK.beta * CAR[K,J])
                dCAR[J,K] = (- (popJ.beta + popK.gamma) * CAR[J,K]
                                + popK.beta * CAA[J,K]
                                + popJ.alpha*popJ.F(B[J]) * CSR[J,K]
                                + popJ.alpha*popJ.dF(B[J]) * S[J] * CRB[K,J])
        f[2*p : 2*p + round(p*(p+1)/2)] = dCAA[np.triu_indices(p)]
        f[2*p + round(p*(p+1)/2) : 2*p + p*(p+1)] = dCRR[np.triu_indices(p)]
        f[2*p + p*(p+1) :] = dCAR.flatten()
        return np.array(f, float)

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        p = len(self.config.network.populations)
        return p * (2*p + 3)


class _TaylorExtendedSystemOne(TaylorExtendedSystem):
    """Special case of `TaylorExtendedSystem` for one population.

    Special case of `TaylorExtendedSystem` to use when the network has a single
    population. It is different from this class only in that it uses a simpler
    implementation of the vector field, and its implements the Jacobian matrix.

    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._pop = self.config.network.populations[0]

    def jac(self, state):
        """Jacobian matrix of the vector field. 

        See `DynamicalSystem.jac` for details.
        """
        A, R, CAA, CRR, CAR = state[0], state[1], state[2], state[3], state[4]
        S = 1 - A - R
        c = self.config.network.c[0,0]
        B = c * A + self.config.Q[0]
        F = self._pop.F(B)
        dF = self._pop.dF(B)
        ddF = self._pop.ddF(B)
        dddF = self._pop.dddF(B)
        j = np.zeros((5,5))
        j[0,0] = (-self._pop.beta - self._pop.alpha*F 
                    + self._pop.alpha*dF * c * S 
                    - self._pop.alpha*ddF * c**2 * (CAA + CAR) 
                    + self._pop.alpha/2 * (-ddF + S*dddF*c) * c**2 * CAA )
        j[0,1] = (-self._pop.alpha*F 
                    - self._pop.alpha/2*ddF * c**2*CAA )
        j[0,2] = (-self._pop.alpha/2*ddF * c**2
                    - self._pop.alpha*dF * c )
        j[0,3] = 0
        j[0,4] = -self._pop.alpha*dF * c
        j[1,0] = self._pop.beta
        j[1,1] = -self._pop.gamma
        j[1,2] = j[1,3] = j[1,4] = 0
        j[2,0] = (-2*self._pop.alpha*dF*c * CAA 
                    - 2*self._pop.alpha*dF*c * CAR 
                    + 2*self._pop.alpha * (-dF + ddF*c*S) * c * CAA )
        j[2,1] = -2*self._pop.alpha*dF * c * CAA
        j[2,2] = (-2 * (self._pop.beta + self._pop.alpha*F) 
                    + 2*self._pop.alpha*dF * c * S )
        j[2,3] = 0
        j[2,4] = -2*self._pop.alpha*F
        j[3,0] = j[3,1] = j[3,2] = 0
        j[3,3] = -2*self._pop.gamma
        j[3,4] = 2*self._pop.beta
        j[4,0] = (-self._pop.alpha*dF * c * CAR 
                    - self._pop.alpha*dF * c * CRR 
                    + self._pop.alpha * (-dF + ddF*c*S) * c * CAR )
        j[4,1] = -self._pop.alpha*dF*c * CAR
        j[4,2] = self._pop.beta
        j[4,3] = -self._pop.alpha*F
        j[4,4] = (-(self._pop.beta + self._pop.gamma + self._pop.alpha*F) 
                    + self._pop.alpha*dF* c * S )
        return np.array(j, float)

    def vector_field(self, state):
        """Vector field of the extended Wilson--Cowan model.

        See `DynamicalSystem.vector_field` for details.
        """
        A, R, CAA, CRR, CAR = state[0], state[1], state[2], state[3], state[4]
        S = 1 - A - R
        c = self.config.network.c[0,0]
        B = c * A + self.config.Q[0]
        F = self._pop.F(B)
        dF = self._pop.dF(B)
        ddF = self._pop.ddF(B)
        f = [0, 0, 0, 0, 0]
        f[0] = (- self._pop.beta*A + self._pop.alpha*F * S 
                - self._pop.alpha*dF * c * (CAA + CAR)
                + self._pop.alpha/2*ddF * c**2 * S * CAA )
        f[1] = - self._pop.gamma*R + self._pop.beta*A
        f[2] = (- 2*(self._pop.beta + self._pop.alpha*F) * CAA 
                - 2*self._pop.alpha*F*CAR 
                + 2*self._pop.alpha*dF * c* S * CAA )
        f[3] = - 2*self._pop.gamma*CRR + 2*self._pop.beta*CAR
        f[4] = (- (self._pop.beta + self._pop.gamma 
                + self._pop.alpha*F) * CAR + self._pop.beta*CAA 
                - self._pop.alpha*F*CRR 
                + self._pop.alpha*dF * c * S * CAR )
        return np.array(f, float)


class ExtendedSystem(DynamicalSystem):
    """Dynamical system for the extended Wilson--Cowan model.

    Specializes `DynamicalSystem` for the extended Wilson--Cowan model,
    obtained from the closure based on sigmoid functions. Here the refractory
    state and the covariances between fractions of populations are included.
    For *p* populations, a state of this system has the form
    \\[
        (\\begin{aligned}[t]
            & A_1, A_2, ..., A_p, \\\\
            & R_1, R_2, ..., R_p, \\\\
            & \\mathrm{C}_{AA}^{11}, \\mathrm{C}_{AA}^{12}, ..., 
                \\mathrm{C}_{AA}^{1p}, \\mathrm{C}_{AA}^{22}, ..., 
                \\mathrm{C}_{AA}^{2p}, \\mathrm{C}_{AA}^{33}, ..., 
                \\mathrm{C}_{AA}^{3p}, ..., \\mathrm{C}_{AA}^{pp}, \\\\
            & \\mathrm{C}_{RR}^{11}, \\mathrm{C}_{RR}^{12}, ..., 
                \\mathrm{C}_{RR}^{1p}, \\mathrm{C}_{RR}^{22}, ..., 
                \\mathrm{C}_{RR}^{2p}, \\mathrm{C}_{RR}^{33}, ..., 
                \\mathrm{C}_{RR}^{3p}, ..., \\mathrm{C}_{RR}^{pp}, \\\\
            & \\mathrm{C}_{AR}^{11}, \\mathrm{C}_{AR}^{12}, ..., 
                \\mathrm{C}_{AR}^{1p}, \\mathrm{C}_{AR}^{21}, ...,
                \\mathrm{C}_{AR}^{2p}, ..., \\mathrm{C}_{AR}^{p1}, 
                \\mathrm{C}_{AR}^{p2}, ..., \\mathrm{C}_{AR}^{pp})
        \\end{aligned}
    \\]
    where \\(A_J\\) and \\(R_J\\) are respectively the expectations of the
    active and refractory fractions of the *J*th population, in the order given
    by the list of populations in the configuration's network, and
    \\(\\mathrm{C}_{XY}^{JK}\\) is the covariance between the fractions
    \\(X_J\\) and \\(Y_K\\), where \\(X\\) and \\(Y\\) stand for random
    variables associated with active or refractory fractions.

    The Jacobian matrix is not implemented for this system.

    """

    def vector_field(self, state):
        """Vector field of the extended Wilson--Cowan model.

        See `DynamicalSystem.vector_field` for details.
        """
        A = state[: (p := len(self.config.network.populations))]
        R = state[p : 2*p]
        S = 1 - A - R
        CAA = _internals._unflat_scalar_triangle(
                    state[2*p : 2*p + round(p*(p+1)/2)])
        CRR = _internals._unflat_scalar_triangle(
                    state[2*p + round(p*(p+1)/2) : 2*p + p*(p+1)])
        CAR = (state[2*p + p*(p+1) :]).reshape((p,p))
        CAS = - CAA - CAR
        CRS = - CRR - CAR.transpose()
        B = self.config.Q.copy()
        VarB = np.zeros(p)
        CAB = np.zeros((p,p))
        CRB = np.zeros((p,p))
        CSB = np.zeros((p,p))
        for J, K in np.ndindex((p,p)):
            B[J] += self.config.network.c[J,K] * A[K]
            for L in range(p):
                VarB[J] += (self.config.network.c[J,K] 
                            * self.config.network.c[J,L] * CAA[K,L])
                CAB[J,K] += self.config.network.c[K,L] * CAA[J,L]
                CRB[J,K] += self.config.network.c[K,L] * CAR[L,J]
                CSB[J,K] += self.config.network.c[K,L] * CAS[L,J]
        f = np.zeros(self.dim)
        dCAA = np.zeros((p,p))
        dCRR = np.zeros((p,p))
        dCAR = np.zeros((p,p))
        for J, popJ in enumerate(self.config.network.populations):
            SG = np.where(S[J] == 0, 0,
                          S[J] * popJ.G(B[J] + CSB[J,J]/S[J], VarB[J]))
            f[J] = (- popJ.beta * A[J] + popJ.alpha * SG)
            f[J+p] = - popJ.gamma * R[J] + popJ.beta * A[J]
            for K, popK in enumerate(self.config.network.populations):
                if J <= K:
                    dCAA[J,K] = (- (popJ.beta + popK.beta) * CAA[J,K]
                                    + popJ.alpha * popJ.H(A[K], S[J], B[J],
                                        CAS[K,J], CAB[K,J], CSB[J,J], VarB[J])
                                    + popK.alpha * popK.H(A[J], S[K], B[K],
                                        CAS[J,K], CAB[J,K], CSB[K,K], VarB[K]))
                    dCRR[J,K] = (- (popJ.gamma + popK.gamma) * CRR[J,K]
                                  + popJ.beta * CAR[J,K] + popK.beta * CAR[K,J])
                dCAR[J,K] = (- (popJ.beta + popK.gamma) * CAR[J,K]
                                + popK.beta * CAA[J,K]
                                + popJ.alpha * popJ.H(R[K], S[J], B[J],
                                    CRS[K,J], CRB[K,J], CSB[J,J], VarB[J]))
        f[2*p : 2*p + round(p*(p+1)/2)] = dCAA[np.triu_indices(p)]
        f[2*p + round(p*(p+1)/2) : 2*p + p*(p+1)] = dCRR[np.triu_indices(p)]
        f[2*p + p*(p+1) :] = dCAR.flatten()
        return np.array(f, float)

    def _get_dimension(self):
        """Get the dimension of the dynamical system."""
        p = len(self.config.network.populations)
        return p * (2*p + 3)


class _BifurcationFinder:
    """Interface to find a bifurcation.

    This class provides an interface to find bifurcations with respect to
    epsilon in the mixed system that describes the transition between the
    classical Wilson--Cowan system and its extension with refractory state.

    Parameters
    ----------
    system : MixedSystem
        Dynamical system for which to find a bifurcation.
    state : array_like
        Initial guess for the state at which there is a bifurcation.

    Attributes
    ----------
    system : MixedSystem
        Dynamical system for which to find a bifurcation.
    steady_state : array_like
        Steady state at which a bifurcation occurs.

    """

    def __init__(self, system, state):
        self.system = system
        self.steady_state = self.system.find_equilibrium_near(state)

    def run(self, precision):
        """Run the bifurcation finder.
        
        Parameters
        ----------
        precision : int
            Precision desired on the value of epsilon at the bifurcation, as
            a number of significant digits.

        Returns
        -------
        float
            Value of epsilon at which the bifurcation occurs, or `None` if no
            bifurcation is found.
        array_like
            Eigenvalues of the Jacobian matrix at the bifurcation point, or
            `None` if no bifurcation is found.
        """
        original_epsilon = self.system.epsilon
        emin, emax = 0, 1
        self.signs_at_zero = self._get_eigenvals_signs_at_zero()
        for j in range(precision):
            emin, emax = self._update_epsilon_bounds(emin, emax)
            if emin is None:
                return None, None
        e = (emax + emin) / 2
        self.system.epsilon = e
        eigenvals = self.system.get_eigenvals_at(self.steady_state)
        self.system.epsilon = original_epsilon
        return e, eigenvals

    def _have_signs_changed(self, eigenvals):
        """Compare signs of eigenvalues' real parts to those at epsilon = 0."""
        for eig, sign in zip(eigenvals, self.signs_at_zero):
            if np.sign(eig.real) != sign:
                return True
        return False

    def _get_eigenvals_signs_at_zero(self):
        """Get signs of eigenvalues' real parts at `state` for epsilon = 0."""
        WC_system = WilsonCowanSystem(self.system.config)
        eigenvals = WC_system.get_eigenvals_at(self.steady_state)
        A_components_signs = [np.sign(np.real(eig)) for eig in eigenvals]
        R_components_signs = [-1 for sign in A_components_signs]
        signs = np.concatenate((A_components_signs, R_components_signs))
        return signs

    def _update_epsilon_bounds(self, emin, emax):
        """Update the bounds between which there is a bifurcation.

        Assuming the dynamical system undergoes a bifurcation with respect to
        epsilon between `emin` and `emax`, increase the precision of these
        bounds by one significant digit.

        Parameters
        ----------
        emin, emax : float
            Bounds between which the system undergoes a bifurcation with
            respect to epsilon.

        Returns
        -------
        emin, emax : array_like
            Bounds analogous to those given but with increased precision, or
            `None` if there is no bifurcation point between the given bounds.
        """
        epsilon = np.linspace(emin, emax, 11)
        p = len(self.system.config.network.populations)
        for j, e in enumerate(epsilon):
            if e == 0:
                continue
            self.system.epsilon = e
            eigenvals = self.system.get_eigenvals_at(self.steady_state)
            if self._have_signs_changed(eigenvals):
                emin = epsilon[j-1]
                emax = epsilon[j]
                break
        else:
            return None, None
        return emin, emax


SYSTEM_CLASSES = {'mean-field': MeanFieldSystem,
                  'wilson-cowan': WilsonCowanSystem,
                  'mixed': MixedSystem,
                  'taylor': TaylorExtendedSystem,
                  'extended': ExtendedSystem}
"""Mapping between keywords and `DynamicalSystem` subclasses."""


def get_system(config, system_name, **kwargs):
    """Define a dynamical system from a configuration.

    Define a dynamical system from the parameters in the given configuration.
    The system can be chosen from a given list of systems related to the
    Wilson--Cowan model, or more generally from any system given in
    `SYSTEM_CLASSES`.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration associated with the dynamical system.
    system_name : str
        Decides which type of dynamical system to return. The following values
        are accepted by default.

         - `'mean-field'`: the Wilson--Cowan's model with refractory state.
         - `'wilson-cowan'`: an equivalent to the original Wilson--Cowan model.
         - `'mixed'`: the mean field one, but with an additional parameter
           multiplying the refractory states' derivates.
         - `'taylor'`: the extended Wilson--Cowan model with the closure
           resulting from a second-order Taylor approximation.
         - `'extended'`: the extended Wilson--Cowan model with the closure
           based on sigmoid functions.

    **kwargs
        Keywords arguments passed to the class constructor.

    Returns
    -------
    DynamicalSystem
        Dynamical system initialized according to the given configuration. It
        is always a *subclass* of `DynamicalSystem` that depends on the chosen
        system.

    Raises
    ------
    popnet.exceptions.PopNetError
        If `system_name` is given a non-valid value.
    TypeError
        If `config` is not a `popnet.structures.Configuration` instance.
    """
    if system_name not in SYSTEM_CLASSES:
        raise PopNetError(f'Unknown dynamical system {system_name}. Valid '
                          f'values are {tuple(SYSTEM_CLASSES.keys())}.')
    return SYSTEM_CLASSES[system_name](config, **kwargs)

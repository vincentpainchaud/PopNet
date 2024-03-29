"""Data structures to represent biological neural networks.

This modules defines several classes which are used as data structures to
represent biological neural networks and configurations used in numerical
experiments. The hierarchy of the module's classes is given in the
[Classes And Hierarchy](#classes-and-hierarchy) section below.

Classes and hierarchy
---------------------

The important classes of the module are summarized below. The indentation
follows the hierarchy. 

 - `Population` : Represent a population of biological neurons.
 - `Network` : Represent a network split into populations.
    - `MicroNetwork` : Represent a network, including its microscopic structure.
 - `Configuration` : A complete configuration to perform a numerical experiment.
     - `ConfigurationOne` : A configuration with a network of one population.
     - `MicroConfiguration` : A configuration with a `MicroNetwork`.
     - `MicroConfigurationOne` : A merge of the two above classes.

"""

import re
import ast
import numpy as np
from copy import deepcopy
from warnings import warn
from scipy.special import expit, logit

from .exceptions import *
from . import _internals


class Population:
    """Represent a population of biological neurons.

    This class is used to describe a population of biological neurons. It
    allows to easily define parameters on a population, such as a threshold and
    characteristic transition rates.

    Parameters
    ----------
    name : str
        Name of the population.
    ID : str, optional
        ID of the population. It must be a single character. If it is not
        given, the last character of `name` is used if it is a number, and
        else the first one is used.
    size : int, optional
        Number of neurons of the population. It must be positive. Defaults to
        `None`, in which case no size is defined for the population.

    Attributes
    ----------
    name : str
        Name given to the population. See `Population.name`.
    ID : str
        ID of the population. See `Population.ID`.
    size : int
        Size of the population. See `Population.size`.
    alpha : float
        Characteristic transition rate from sensitive to active. See
        `Population.alpha`.
    beta : float
        Mean transition rate from active to refractory. See `Population.beta`.
    gamma : float
        Mean transition rate from refractory to sensitive. See
        `Population.gamma`.
    theta : float
        Threshold. See `Population.theta`.
    scale_alpha : float
        Scale of the characteristic transition rates from sensitive to active.
        See `Population.scale_alpha`.
    scale_beta : float
        Scale of transition rates from active to refractory. See
        `Population.scale_beta`.
    scale_gamma : float
        Scale of transition rates from refractory to sensitive. See
        `Population.scale_gamma`.
    scale_theta : float
        Scale of thresholds. See `Population.scale_theta`.

    """

    def __init__(self, name, ID=None, size=None):
        self.name = name
        if ID is None:
            ID = self._default_ID()
        self.ID = ID
        self.size = size

        self._means = {'alpha': 1., 'beta': 1., 'gamma': 1., 'theta': 0.}
        self._scales = {'alpha': 0., 'beta': 0., 'gamma': 0., 'theta': 1.}
        self._update_means()
        self._update_scales()

    def __str__(self):
        string = f'{self.name}'
        if self.size is not None:
            string += f' - {self.size} neurons'
        for mean in self._means:
            string += f'\n{mean:>11} = {self._means[mean]}'
        for scale in self._scales:
            scale_name = 'scale ' + str(scale)
            string += f'\n{scale_name:>11} = {self._scales[scale]}'
        return string

    @property
    def name(self):
        """Name of the population. 

        Name given to the population to identify it in a network. It has to be
        a string. When setting a new name, if the population's ID was the
        default one, then the ID is updated according to the default rule with
        the new name: the ID is either taken to be the last character of the
        name if it is a number, or else it is taken to be the first character
        of the name.

        A population's name should never contain the string " - ", because this
        could lead to unexpected behavior when saving and loading data. An error
        is raised if a name containing this string is set.
        """
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError('A population\'s name must be a string.')
        try:
            assert self.ID == self._default_ID()
        except (AttributeError, AssertionError):
            pass
        else:
            self.ID = self._default_ID(new_name)
        if ' - ' in new_name:
            raise PopNetError('A population\'s name cannot contain \' - \'.')
        self._name = new_name

    @property
    def ID(self):
        """ID of the population. 

        ID given to the population. It must be a single character, else an
        error is raised when setting it. The ID is used as a subscript to
        identify state variables. 
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if len(str(new_ID)) != 1:
            raise PopNetError('A population\'s ID must be a single character.')
        self._ID = str(new_ID)

    @property
    def size(self):
        """Number of neurons in the population. 

        Number of neurons in the population. When it is defined, it must be a
        positive integer.
        """
        return self._size

    @size.setter
    def size(self, new_value):
        if new_value is None:
            self._size = None
            return
        try:
            new_value = int(new_value)
        except TypeError as error:
            raise TypeError('A population\'s size must be a number.') from error
        else:
            if not new_value > 0:
                raise ValueError('A population\'s size must be positive.')
        self._size = new_value

    @property
    def alpha(self):
        """Characteristic `alpha` transition rate in the population.

        Characteristic transition rate from sensitive to active in the
        population. From the macroscopic point of view, this parameter can
        be interpreted as the mean value of the supremal (i.e., with infinite
        input) transition rates from sensitive to active over the neurons of
        the population.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, new_value):
        self._means['alpha'] = float(new_value)
        self._alpha = float(new_value)

    @property
    def beta(self):
        """Mean `beta` transition rate in the population.

        Mean value of the transition rates from active to refractory in the
        population.
        """
        return self._beta

    @beta.setter
    def beta(self, new_value):
        self._means['beta'] = float(new_value)
        self._beta = float(new_value)

    @property
    def gamma(self):
        """Mean `gamma` transition rate in the population.

        Mean value of the transition rates from refractory to sensitive in the
        population.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, new_value):
        self._means['gamma'] = float(new_value)
        self._gamma = float(new_value)

    @property
    def theta(self):
        """Mean threshold in the population. 

        Mean value of the thresholds of activation in the population. This
        parameter can have two interpretations, depending on the stochastic
        process that is assumed to rule the evolution of the network's state.
        
         1. When the activation rate of a neuron is a step function of its
            input, `theta` is the mean value of the individual thresholds
            of neurons in the population.
         2. When the activation rate of a neuron is a sigmoid function of
            its input, `theta` is the inflexion point of this sigmoid
            function.
        """
        return self._theta

    @theta.setter
    def theta(self, new_value):
        self._means['theta'] = float(new_value)
        self._theta = float(new_value)

    @property
    def scale_alpha(self):
        """Scaling factor of the `alpha` transition rate in the population.

        Scaling factor of the distribution of the supremal (i.e., with infinite
        input) transition rates from sensitive to active in the population, 
        which is assumed to be a logistic distribution.
        """
        return self._scale_alpha

    @scale_alpha.setter
    def scale_alpha(self, new_value):
        self._scales['alpha'] = float(new_value)
        self._scale_alpha = float(new_value)

    @property
    def scale_beta(self):
        """Scaling factor of the `beta` transition rate in the population.

        Scaling factor of the distribution of the transition rates from
        active to refractory in the population, which is assumed to be a
        logistic distribution.
        """
        return self._scale_beta

    @scale_beta.setter
    def scale_beta(self, new_value):
        self._scales['beta'] = float(new_value)
        self._scale_beta = float(new_value)

    @property
    def scale_gamma(self):
        """Scaling factor of the `gamma` transition rate in the population.

        Scaling factor of the distribution of the transition rates from
        refractory to sensitive in the population, which is assumed to be a
        logistic distribution. 
        """
        return self._scale_gamma

    @scale_gamma.setter
    def scale_gamma(self, new_value):
        self._scales['gamma'] = float(new_value)
        self._scale_gamma = float(new_value)

    @property
    def scale_theta(self):
        """Scaling factor of the thresholds in the population.

        Scaling factor associated with the thresholds in the population. This
        parameter can have two interpretations, depending on the stochastic
        process that is assumed to rule the evolution of the network's state.
        
         1. When the activation rate of a neuron is a step function of its
            input, `scale_theta` is the scaling factor of the distribution
            of the thresholds in the population, which is assumed to be a
            logistic distribution.
         2. When the activation rate of a neuron is a sigmoid function of
            its input, `scale_theta` is the scaling parameter of this
            sigmoid function.
        """
        return self._scale_theta

    @scale_theta.setter
    def scale_theta(self, new_value):
        self._scales['theta'] = float(new_value)
        self._scale_theta = float(new_value)

    def copy(self, name, ID=None):
        """Copy the population.

        Return a copy of the population with a new name and ID.

        Parameters
        ----------
        name : str
            Name to give to the new population.
        ID : str, optional
            ID to give to the new population. Defaults to `None`, in which case
            a default one is taken from the name.

        Returns
        -------
        Population
            The copied population.
        """
        other = deepcopy(self)
        other.name = name
        if ID is None:
            ID = other._default_ID()
        other.ID = ID
        return other

    def F(self, y):
        """Macroscopic response function of the population.

        Macroscopic response function of the population. It is always a
        logistic function whose inflexion point is at `Population.theta`, with
        scaling factor `Population.scale_theta`.

        This function can have two interpretations, depending on the stochastic
        process that is assumed to rule the evolution of the network's state.
        
         1. When the activation rate of a neuron is a step function of its
            input, `F` is the cumulative distribution function of the
            thresholds in the population.
         2. When the activation rate of a neuron is a sigmoid function of
            its input, this sigmoid function is the product of
            `Population.alpha` and `F`.
        """
        return expit((y - self.theta) / self.scale_theta)

    def dF(self, y):
        """First derivative of `Population.F`."""
        return 1 / self.scale_theta * self.F(y) * (1 - self.F(y))

    def ddF(self, y):
        """Second derivative of `Population.F`."""
        return 1 / self.scale_theta**2 * ( 
                                self.F(y) * (1 - self.F(y)) * (1 - 2*self.F(y)))

    def dddF(self, y):
        """Third derivative of `Population.F`."""
        return 1 / self.scale_theta**3 * (
               self.F(y) * (1 - self.F(y)) * (1 - 6*self.F(y) + 6*self.F(y)**2))

    def Finv(self, y):
        """Inverse of `Population.F`."""
        return self.theta + self.scale_theta * logit(y)

    def g(self, x, y):
        """Scaling factor added to `Population.F` to define `Population.G`."""
        return np.where(x == self.theta, y / (4 * self.scale_theta**2),
                        (y / (2 * (self.theta - x) * self.scale_theta)
                                * (1 - 2*self.F(x))))

    def G(self, x, y):
        """Rescaled version of `Population.F`."""
        return self.F((x + self.theta * self.g(x,y)) / (1 + self.g(x,y)))

    def H(self, X, S, B, CXS, CXB, CSB, VarB):
        """Approximation of the covariance between *X* and *SF(B)*.

        Function used to approximate the covariance between *X* and *SF(B)* in
        terms of the expectations and covariances of *X*, *S* and *B* and of the
        variance of *B*. Here *F* is `Population.F`. This method is mainly
        intended to be used in the dynamical system for the extended
        Wilson--Cowan model.
        """
        return np.where(S == 0, 0, 
                        np.where(X == 0, 0,
                                 ((X*S + CXS) * self.G(B + CSB/S + CXB/X, VarB)
                                    - X*S * self.G(B + CSB/S, VarB))))

    def set_means(self, **new_values):
        """Set the means of population parameters.

        Assign new values to some of the means of the population's parameters:
        `Population.alpha`, `Population.beta`, `Population.gamma` and
        `Population.theta`.

        Parameters
        ----------
        **new_values
            New values to assign to means of valid population parameters. New
            values are passed as key-value pairs, where the key is the
            parameter and the value is the new value to assign.

        Raises
        ------
        KeyError
            If non-valid parameters are requested.
        """
        for key in new_values:
            if key not in self._means:
                raise KeyError(f'{key} is not a valid population parameter.')
            self._means[key] = float(new_values[key])
        self._update_means()

    def set_random_rates(self, rates=None, distribution='exponential', **kwargs):
        """Randomly set the transition rates from a given distribution.

        Choose a random value for the transition rates from a given distribution
        family with given parameters. The random value is generated using a
        [`Generator`](https://31c8.short.gy/np-random-generator) instance from
        NumPy's `random` module, and keyword arguments can be passed to the
        `Generator`'s method used to generate the random values.

        Parameters
        ----------
        rates : list or tuple of str, or str, optional
            The transition rates to be chosen randomly. It must contain only
            valid rates in `alpha`, `beta` or `gamma`.
        distribution : {'uniform', 'exponential'}, optional
            The distribution family used to choose a value for the rates.
            Defaults to `'uniform'`.
        **kwargs
            Keyword arguments to be passed to the method of `Generator`
            corresponding to the correct distribution family.

        Raises
        ------
        TypeError
            If `rates` is not a list, tuple, or str.
        KeyError
            If the strings given in `rates` are not valid transition rates.
        NotImplementedError
            If the requested distribution family is not implemented.
        """
        if rates is None:
            rates = ['alpha', 'beta', 'gamma']
        if isinstance(rates, str):
            rates = [rates]
        if not isinstance(rates, (list, tuple)):
            raise TypeError('Transition rates must be given as a list or tuple '
                            f'of strings, not as a {type(rates)} object.')
        for rate in rates:
            if rate not in ['alpha', 'beta', 'gamma']:
                raise KeyError(f'{rate} is not a valid transition rate.')
            rng = np.random.default_rng()
            if distribution == 'uniform':
                self._means[rate] = rng.uniform(**kwargs)
                continue
            elif distribution == 'exponential':
                self._means[rate] = rng.exponential(**kwargs)
                continue
            raise NotImplementedError(f'No distribution \'{distribution}\' '
                                      'available to set transition rates.')
        self._update_means()

    def set_random_threshold(self, distribution='uniform', **kwargs):
        """Randomly set the threshold from a given distribution.

        Choose a random value for the threshold from a given distribution family
        with given parameters. The random value is generated using a
        [`Generator`](https://31c8.short.gy/np-random-generator) instance from
        NumPy's `random` module, and keyword arguments can be passed to the
        `Generator`'s method used to generate the random values.

        Parameters
        ----------
        distribution : {'uniform'}, optional
            The distribution family used to choose a value for the threshold.
            Defaults to `'uniform'`, which is for now the only implemented
            distribution.
        **kwargs
            Keyword arguments to be passed to the method of `Generator`
            corresponding to the correct distribution family.

        Raises
        ------
        NotImplementedError
            If the requested distribution family is not implemented.
        """
        if distribution == 'uniform':
            self.theta = np.random.default_rng().uniform(**kwargs)
            return
        raise NotImplementedError(f'No distribution \'{distribution}\' '
                                  'available to set a threshold.')

    def set_scales(self, **new_values):
        """Set the scales of population parameters.

        Assign new values to some of the scales of the population's parameters:
        `Population.alpha`, `Population.beta`, `Population.gamma` and
        `Population.theta`.

        Parameters
        ----------
        **new_values
            New values to assign to scales of valid population parameters. New
            values are passed as key-value pairs, where the key is the
            parameter and the value is the new value to assign.

        Raises
        ------
        KeyError
            If non-valid parameters are requested.
        """
        for key in new_values:
            if key not in self._scales:
                raise KeyError(f'{key} is not a valid population parameter.')
            self._scales[key] = float(new_values[key])
        self._update_scales()

    def _default_ID(self, name=None):
        """Get a default ID based on `name`.

        Get a default ID based on the name `name`. If `name` ends with a number,
        this number is returned. Else, the first letter of `name` is returned.

        Parameters
        ----------
        name : str, optional
            Name from which to get an ID. Defaults to `None`, in which case the
            `name` attribute is used instead.

        Returns
        -------
        str
            The said default ID.
        """
        if name is None:
            name = self.name
        assert isinstance(name, str), '\'name\' argument must be a string.'
        try:
            no = int(name[-1])
        except Exception:
            return name[0]
        else:
            return str(no)

    @classmethod
    def _load(cls, lines):
        """Load a population.

        Load a population's parameters from a list of strings.

        Parameters
        ----------
        lines : list of str
            Strings from which the parameters are to be set. It is expected to
            be the lines of a string representation of a `Population` instance.

        Returns
        -------
        Population
            The loaded population.

        Raises
        ------
        KeyError
            If `string` contains assignments to non-valid parameters.
        popnet.exceptions.FormatError
            If `string` does not have the expected format. 
        """
        lines = [line.replace('\n', '') for line in lines]
        name_line = lines[0].split(' - ')
        pop = Population(name_line[0].strip())
        if len(name_line) == 2:
            try:
                pop.size = name_line[-1].strip().split()[0]
            except ValueError as err:
                raise PopNetError('An unexpected error occurred when loading '
                                  'data for a population. It might be due to a '
                                  'name containing the string " - ".') from err
        for line in lines[1:]:
            # Take the part of the line specifying the parameter to set.
            param_spec = line[:11].strip().split()
            if (n := len(param_spec)) == 0:
                # If there is nothing on the line, continue to the next one.
                continue
            if n == 1:
                # If there is a single word, it should give a parameter's mean.
                param = param_spec[0]
                if param not in pop._means:
                    raise KeyError(f'{param} is not a valid parameter.')
                pop._means[param] = float(line[14:])
            elif n == 2:
                # If there are two words, it should give a parameter's scale.
                scale = param_spec[0]
                param = param_spec[1]
                if scale != 'scale' or param not in pop._scales:
                    raise KeyError(f'{scale} {param} is not a valid parameter.')
                pop._scales[param] = float(line[14:])
            else:
                # If there are more than two words, something went wrong...
                raise FormatError('The given string cannot be used to define '
                                  'the parameters of a population.')
        pop._update_means()
        pop._update_scales()
        return pop

    def _update_means(self):
        """Update the parameters' means according to `_means` values.

        Update the attributes `alpha`, `beta`, `gamma` and `theta` according to
        the values of the corresponding entries of `._means`. It is intended to
        be used internally in other methods when setting new values to
        parameters, to ensure that the values in `_means` are consistent
        with the values of the corresponding attributes.
        """
        self.alpha = self._means['alpha']
        self.beta  = self._means['beta']
        self.gamma = self._means['gamma']
        self.theta = self._means['theta']

    def _update_scales(self):
        """Update the parameters' scales according to `_scales` values.

        Update the attributes `scale_alpha`, `scale_beta`, `scale_gamma` and
        `scale_theta` according to the values of the corresponding entries of
        `_scales`. It is intended to be used internally in other methods when
        setting new values to parameters, to ensure that the values in `_scales`
        are consistent with the values of the corresponding attributes.
        """
        self.scale_alpha = self._scales['alpha']
        self.scale_beta  = self._scales['beta']
        self.scale_gamma = self._scales['gamma']
        self.scale_theta = self._scales['theta']


class Network:
    """Represent a biological neural network from a macroscopic point of view.

    This class is used to represent biological neural networks split into
    populations. The purpose of this class is to have a consistent interface
    to define, modify, save, or load the parameters of a network.

    Parameters
    ----------
    ID : str
        ID of the network.
    populations : list or tuple of Population or Population
        Defines the populations that constitute the network. It can be given as
        a `Population` instance to make a network with a single population. 

    Attributes
    ----------
    ID : str
        ID of the network. See `Network.ID`.
    populations : tuple of Population
        Populations of the network. See `Network.populations`.
    c : array_like
        Connection matrix. See `Network.c`.
    scale_c : array_like
        Scale of connection weights. See `Network.scale_c`.

    Raises
    ------
    TypeError
        If `populations` cannot be converted to a tuple of `Population`
        instances.

    """

    def __init__(self, ID, populations):
        try:
            self._populations = tuple(populations)
        except TypeError:
            self._populations = (populations,)
        if not all(isinstance(pop, Population) for pop in self._populations):
            raise TypeError('The \'populations\' attribute of a \'Network\' ins'
                            'tance must be a tuple of \'Population\' instances.')
        self.ID = ID
        self.c = np.ones((p := len(self.populations), p))
        self.scale_c = np.zeros((p, p))

    def __str__(self):
        string = f'Network {self.ID}\n\n'
        for pop in self.populations:
            string += str(pop)
            string += '\n\n'
        string += f'Connection matrix:\n{self.c}\n\n'
        string += f'Connection scales matrix:\n{self.scale_c}'
        return string

    @property
    def ID(self):
        """ID of the network.

        ID given to the network. Its first character has to be the number of
        populations of the network, else an error is raised when setting it.
        The ID is used to name files when saving the network parameters.
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if not isinstance(new_ID, str):
            raise TypeError('The network\'s ID must be a string.')
        if int(new_ID[0]) != len(self.populations):
            raise PopNetError('The first character of the network\'s ID must '
                              'be its number of populations.')
        self._ID = new_ID

    @property
    def populations(self):
        """Populations of the network.

        Tuple containing the populations of the network, given as `Population`
        instances. It is set at initialization, but it cannot be reset nor
        deleted afterwards.
        """
        return self._populations

    @property
    def c(self):
        """Connection matrix of the network.

        Describes the weights of connections between populations of the network.
        The exact relation to the weights of links between individual neurons of
        the network is described in the [Notes](#network-c-notes) section below.
        It has to be a square matrix, but it can be given as a float if the
        network has only one population.

        Notes {#network-c-notes}
        -----
        For clarity, let *J* and *K* be the *j*th and *k*th populations of the
        network respectively, in the order given by `Network.populations`.
        Then, the element `c[j,k]` of `c` describes the link *from K
        to J*. From the microscopic point of view, it is the product of the size
        of *K* with the mean value of the weights of links from neurons of *K*
        to neurons of *J*. 
        """
        return self._c

    @c.setter
    def c(self, new_c):
        try:
            float_new_c = float(new_c)
        except Exception:
            pass
        else:
            new_c = np.array([[float_new_c]])
        if np.shape(new_c) != (p := len(self.populations), p):
            raise PopNetError('The connection matrix \'c\' must be a square '
                              'array whose size corresponds to the number of '
                              'populations of the network.')
        self._c = np.array(new_c, float)

    @property
    def scale_c(self):
        """Scaling factors of the weights' distributions.

        Scaling factors used to define the weights' distributions, which are
        all assumed to be logistic. The exact relation to the weights of links
        between individual neurons of the network is described in the
        [Notes](#network-scale-c-notes) section below. It has to be a square
        matrix, but it can be given as a float if the network has only one
        population.

        Notes {#network-scale-c-notes}
        -----
        For clarity, let *J* and *K* be the *j*th and *k*th populations of the
        network respectively, in the order given by `Network.populations`.
        Then, the actual scaling factor of the logistic distribution
        of the weights of links from neurons of *K* to neurons of *J* is
        *s*<sub>*JK*</sub>/|*K*|, where *s*<sub>*JK*</sub> is `scale_c[j,k]`
        and |*K*| is the size of *K*.
        """
        return self._scale_c

    @scale_c.setter
    def scale_c(self, new_scale):
        try:
            float_new_scale = float(new_scale)
        except Exception:
            pass
        else:
            new_scale = np.array([[float_new_scale]])
        if np.shape(new_scale) != (p := len(self.populations), p):
            raise PopNetError('The scales of the weights must be a square '
                              'array whose size corresponds to the number of '
                              'populations of the network.')
        self._scale_c = np.array(new_scale, float)

    def copy(self, new_ID):
        """Copy the network.

        Return a copy of the network with a new ID. 

        Parameters
        ----------
        new_ID : str
            ID to give to the new network.

        Returns
        -------
        Network
            The copied network.
        """
        other = deepcopy(self)
        other.ID = new_ID
        return other

    def extend(self, ID):
        """Extend the network.

        Return an extension of the present network, that is, return a new
        network with the same populations and the same connections, but with
        more populations as well.

        The new network has the number of populations given by the first
        character of `ID`. This number should be higher than the number of
        populations of the present network. If it is equal, a copy of the
        network is simply returned.

        Note that the resulting network never has a microscopic structure,
        since the new populations are always defined with no sizes.
        
        Parameters
        ----------
        ID : str
            ID of the new network. Its first character gives the number of
            populations of the new network.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the network is asked to be extended into a smaller network.
        """
        new_p = int(ID[0])
        p = len(self.populations)
        if new_p < p:
            raise PopNetError('Can\'t extend a network into a smaller one.')
        if new_p == p:
            return self.copy()
        pops = [pop.copy(pop.name) for pop in self.populations]
        for j in range(p, new_p):
            pops.append(Population(f'Population {j+1}'))
        net = Network(ID, pops)
        net.c[:p,:p] = self.c
        net.scale_c[:p,:p] = self.scale_c
        return net

    def save(self, folder=None, note=None):
        """Save the network's parameters in a text file.

        Save the string representation of the network in a text file, under the
        name *ID - Network parameters.txt*, where *ID* is the `ID` attribute.

        Parameters
        ----------
        folder : str, optional
            A folder in which the file is saved. If it does not exist in the
            current directory, it is created. Defaults to `None`, in which case
            the file is saved in the current directory.
        note : str, optional
            If given, an additional section "Additional notes:" is written in
            the file, and `note` is written there. 
        """
        filename = _internals._format_filename(folder, self.ID, 
                                               'Network parameters')
        _internals._make_sure_folder_exists(folder)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))
            if note is not None:
                file.write('\n\nAdditional notes:\n')
                file.write(note)

    def set_random_c(self, distribution='uniform', signs=None, **kwargs):
        """Randomly set the connection matrix. 

        Choose random values for entries of the connection matrix from a given
        distribution family with given parameters. The random value is generated
        using a [`Generator`](https://31c8.short.gy/np-random-generator)
        instance from NumPy's `random` module, and keyword arguments can be
        passed to the `Generator`'s method used to generate the random values.

        Parameters
        ----------
        distribution : {'uniform', 'exponential'}, optional
            The distribution family used to choose a value for the threshold. If
            a positive distribution is chosen, the signs of the components of
            `c` are supposed to be fixed by `signs`. Defaults to `'uniform'`. 
        signs : array_like, optional
            A matrix that multiplies the random results. It is intended to be
            used to assign specific signs to the components of `c`. It must be
            a square matrix of -1's and 1's of the same shape as `c`. Defaults
            to `None`, in which case it is replaced by an array of ones.
        **kwargs
            Keyword arguments to be passed to the method of `Generator` 
            corresponding to the correct distribution family. 

        Raises
        ------
        NotImplementedError
            If the requested distribution family is not implemented. 
        """
        shape = (p := len(self.populations), p)
        if signs is None:
            signs = np.ones(shape)
        else:
            signs = np.array(signs, float)
        rng = np.random.default_rng()
        if distribution == 'uniform':
            self.c = signs * rng.uniform(size=shape, **kwargs)
            return
        elif distribution == 'exponential':
            self.c = signs * rng.exponential(size=shape, **kwargs)
            return
        raise NotImplementedError(f'No distribution \'{distribution}\' '
                                  'available to set a connection matrix.')

    def underlying(self):
        """Get the microscopic network underlying the present macroscopic one.

        Return the microscopic network underlying the present macroscopic
        network of populations. The returned network has the same ID, the same
        populations and the same parameters as the present one, but has a
        defined microscopic structure.

        Returns
        -------
        MicroNetwork
            The underlying microscopic network.
        """
        microself = MicroNetwork(self.ID, self.populations)
        microself.c = self.c
        microself.scale_c = self.scale_c
        microself.reset_parameters()
        return microself


class MicroNetwork(Network):
    """Represent a biological neural network from a microscopic point of view.

    `MicroNetwork` extends `Network` to define parameter values for individual
    neurons rather than only defining their mean values and scales by
    populations. It introduces new attributes `MicroNetwork.alpha`,
    `MicroNetwork.beta`, `MicroNetwork.gamma`, `MicroNetwork.theta` and
    `MicroNetwork.W` to get the values of transition rates, thresholds and
    weights of connection for all neurons.

    The initialization is the same as in the base class, except that the
    parameters of individual neurons of the network are also initialized
    automatically: they are randomly generated from the macroscopic parameters
    of the populations. Hence, the size of every population of the network
    must be defined.

    !!! note
        It is important to understand that, even if parameters
        `MicroNetwork.alpha`, `MicroNetwork.beta`, `MicroNetwork.gamma`,
        `MicroNetwork.theta` and `MicroNetwork.W` are generated automatically
        from the corresponding mean values and scaling factors of populations
        at initialization, it does *not* mean that they are automatically
        updated upon update of the mean values or scaling factors, or upon
        change in the size of the network. In order to remain consistent when
        new values are set, parameters should be reset with
        `MicroNetwork.reset_parameters`.

    Raises
    ------
    popnet.exceptions.PopNetError
        If the size of a population is not defined.

    """

    def __init__(self, ID, populations):
        super().__init__(ID, populations)
        if any(pop.size is None for pop in populations):
            raise PopNetError('Cannot define a \'MicroNetwork\' if the sizes '
                              'of its populations are not defined.')
        self.reset_parameters()

    @property
    def alpha(self):
        """Supremal transition rates from sensitive to active.
        
        Array of supremal (i.e., with infinite input) transition rates from
        sensitive to active of all neurons of the network. It cannot be
        manually set nor deleted, but it can be reset with
        `MicroNetwork.reset_parameters`.
        """
        return self._alpha

    @property
    def beta(self):
        """Transition rates from active to refractory.
        
        Array of transition rates from active to refractory of all neurons of
        the network. It cannot be manually set nor deleted, but it can be reset
        with `MicroNetwork.reset_parameters`.
        """
        return self._beta

    @property
    def gamma(self):
        """Transition rates from refractory to sensitive.
        
        Array of transition rates from refractory to sensitive of all neurons
        of the network. It cannot be manually set nor deleted, but it can be
        reset with `MicroNetwork.reset_parameters`.
        """
        return self._gamma

    @property
    def theta(self):
        """Thresholds.
        
        Array of thresholds of all neurons of the network. It cannot be
        manually set nor deleted, but it can be reset with
        `MicroNetwork.reset_parameters`.

        This parameter only has a meaning when activation rates of neurons are
        assumed to be step functions, in which case `theta` gives the
        activation thresholds of neurons. When activation rates of neurons are
        rather assumed to be sigmoid functions, this parameter has no meaning
        (and it is not used when simulating the stochastic process with such
        activation functions).
        """
        return self._theta

    @property
    def W(self):
        """Weight matrix.
        
        Array of weights of connection between neurons of the network. An
        element `W[j,k]` of `W` is the weight of the connection *from* `k` *to*
        `j`. It has to be a real matrix of shape \\(N \\times N\\), where
        \\(N\\) is the size of the network. It cannot be deleted.
        """
        return self._W

    @W.setter
    def W(self, new_value):
        try:
            new_value = np.array(new_value, float)
        except (TypeError, ValueError) as err:
            raise ValueError('A weight matrix must have real entries.') from err
        if new_value.shape != (self.size(), self.size()):
            raise PopNetError('A weight matrix must be square with shape N x N,'
                              ' where N is the size of the network.')
        self._W = new_value

    def reset_parameters(self, params=None):
        """Randomly generate the parameters of the network's neurons.

        Generate the parameters that characterize the neurons of the network.
        All parameters are taken from logistic distributions with means and
        scaling factors consistent with the values given by the populations.
        Since transition rates must be positive, the logistic distributions for
        them are in fact truncated --- see the
        [Notes](#micronetwork-reset-parameters-notes) section below.

        Parameters
        ----------
        params : list or tuple of str or str, optional
            Parameters to be reset. It can contain only valid parameters
            (`'alpha'`, `'beta'`, `'gamma'`, `'theta'` or `'W'`), or be a single
            parameter given as a string. Defaults to `None`, in which case all
            parameters are reset.

        Raises
        ------
        TypeError
            If `params` is neither a list, a tuple nor a string.
        PopNetError
            If an entry of `params` is not a valid population parameter.

        Notes {#micronetwork-reset-parameters-notes}
        -----
        All transition rates `alpha`, `beta` and `gamma` are always positive.
        Hence, when taking samples from logistic distributions to get
        individual values for these pararameters for all neurons of the
        network, the logistic distributions are actually truncated by
        rejecting all negative values and replacing them with other samples.
        """
        valid_params = ('alpha', 'beta', 'gamma', 'theta', 'W')
        if params is None:
            params = valid_params
        if isinstance(params, str):
            params = (params,)
        if not isinstance(params, (list, tuple)):
            raise TypeError('\'params\' must be a list, tuple or string.')
        if any(param not in valid_params for param in params):
            raise PopNetError(f'An entry in {params} is not a valid population '
                              'parameter.')
        rng = np.random.default_rng()

        def sample_rate(rng, mean, scale, size):
            sample = rng.logistic(mean, scale, size=size)
            nsteps = 0
            while np.any(sample < 0) or nsteps > 100:
                sample = np.where(sample < 0, rng.logistic(mean, scale), sample)
                nsteps += 1
            return sample

        if 'alpha' in params:
            self._alpha = np.concatenate(
                        [sample_rate(rng, pop.alpha, pop.scale_alpha, pop.size)
                         for pop in self.populations])
        if 'beta' in params:
            self._beta  = np.concatenate(
                        [sample_rate(rng, pop.beta, pop.scale_beta, pop.size)
                         for pop in self.populations])
        if 'gamma' in params:
            self._gamma = np.concatenate(
                        [sample_rate(rng, pop.gamma, pop.scale_gamma, pop.size)
                         for pop in self.populations])
        if 'theta' in params:
            self._theta = np.concatenate(
                        [rng.logistic(pop.theta, pop.scale_theta, size=pop.size)
                         for pop in self.populations])
        if 'W' in params:
            self._W = np.block([[rng.logistic(
                                    self.c[J,K]/popK.size,
                                    self.scale_c[J,K]/popK.size,
                                    size=(popJ.size, popK.size)) 
                                for K, popK in enumerate(self.populations)] 
                                for J, popJ in enumerate(self.populations)])

    def size(self):
        """Get the size of the network."""
        return np.sum([pop.size for pop in self.populations])

    @property
    def underlying(self):
        raise AttributeError('\'MicroNetwork\' object has no attribute '
                             '\'underlying\'')


class Configuration:
    """Configurations used in numerical experiments.

    This class allows to easily group together all parameters that are needed
    to perform numerical experiments. Although the base class can be used
    with any network of any number of populations, it is better to use the
    `ConfigurationOne` subclass for the case where the network has only one
    population and the `MicroConfiguration` subclass when the network has a
    defined microscopic structure, as more features are available in these
    cases.

    Parameters
    ----------
    network : Network
        Network associated with the configuration.
    ID : str, optional
        ID to associate with the configuration. Defaults to `None`, in which
        case the network's ID is used.
    **kwargs
        Keyword arguments used to initialize other data attributes.

    Attributes
    ----------
    ID : str
        ID of the configuration. See `Configuration.ID`.
    network : Network
        Network associated with the configuration. See `Configuration.network`.
    initial_state : array_like
        Initial state of the network. See `Configuration.initial_state`.
    Q : array_like
        Input in the network. See `Configuration.Q`.
    initial_time, final_time : float
        Times between which the evolution of the network's state is studied.
        See `Configuration.initial_time` and `Configuration.final_time`.
    iterations : int
        Number of iterations in numerical integrations or of time steps in
        chain simulations. See `Configuration.iterations`.
    delta : float
        Time step between two iterations. See `Configuration.delta`.

    Raises
    ------
    TypeError
        If the first argument is not a `Network` instance.
    KeyError
        If a keyword argument is not a valid attribute.

    """

    def __init__(self, network, ID=None, **kwargs):
        if not isinstance(network, Network):
            raise TypeError('The network associated with a configuration '
                            'must be a \'Network\' instance.')
        self._network = network

        if ID is None:
            ID = network.ID
        self.ID = ID

        state_attributes = {'Q': np.zeros(p := len(self.network.populations)), 
                            'initial_state': np.zeros(p * (2*p + 3))}
        float_attributes = {'initial_time': 0., 'final_time': 10.}
        int_attributes = {'iterations': 1000}

        # Here the time attributes have to be initialized without calling the 
        # setter methods, because they reference each other, so they all have to
        # be already defined when a setter is called. 
        for attr in kwargs:
            if attr in state_attributes:
                setattr(self, attr, kwargs[attr])
                state_attributes.pop(attr)
            elif attr in float_attributes:
                setattr(self, '_'+attr, float(kwargs[attr]))
                float_attributes.pop(attr)
            elif attr in int_attributes:
                setattr(self, '_'+attr, int(kwargs[attr]))
                int_attributes.pop(attr)
            else:
                raise KeyError(f'{attr} is not a valid default attribute for '
                               'a \'Configuration\' object.')

        for attr in state_attributes:
            setattr(self, attr, state_attributes[attr])
        for attr in float_attributes:
            setattr(self, '_'+attr, float_attributes[attr])
        for attr in int_attributes:
            setattr(self, '_'+attr, int_attributes[attr])

        self.delta = (self.final_time - self.initial_time) / self.iterations

        A_labels = [f'A[{pop.ID}]' for pop in self.network.populations]
        R_labels = [f'R[{pop.ID}]' for pop in self.network.populations]
        CAA_labels = [f'CAA[{popJ.ID},{popK.ID}]'
                        for J, popJ in enumerate(self.network.populations)
                        for popK in self.network.populations[J:]]
        CRR_labels = [f'CRR[{popJ.ID},{popK.ID}]'
                        for J, popJ in enumerate(self.network.populations)
                        for popK in self.network.populations[J:]]
        CAR_labels = [f'CAR[{popJ.ID},{popK.ID}]'
                        for popJ in self.network.populations
                        for popK in self.network.populations]
        self._variables = (A_labels + R_labels 
                            + CAA_labels + CRR_labels + CAR_labels)

    def __str__(self):
        string = (f'Configuration {self.ID}\n\n'
                  f'Network used: {self.network.ID}\n')
        string += ('\nParameters:\n'
                  f'       ti = {self.initial_time}\n'
                  f'       tf = {self.final_time}\n'
                  f'       \u0394t = {self.delta}\n'
                  f'{self.iterations:>9} iterations\n')
        if self._other_params_string() is not None:
            string += self._other_params_string()
        string += ('\nInput:\n'
                  f'        Q = {self.Q}\n')
        string += '\nInitial state:\n'
        for var, val in zip(self._variables, self.initial_state):
            string += f'{var:>9} = {val}\n'
        return string[:-1] #Remove the last '\n'

    @property
    def ID(self):
        """ID of the configuration.

        ID given to the configuration. It has to be a string that begins with
        the associated network's ID. It is used to name files when saving the
        configuration. Setting its value raises an error if it is not a string
        or if it does not begin with the network's ID.
        """
        return self._ID

    @ID.setter
    def ID(self, new_ID):
        if not isinstance(new_ID, str):
            raise TypeError('The configuration\'s ID must be a string.')
        if new_ID[:len(self.network.ID)] != self.network.ID:
            raise ValueError('The configuration\'s ID must begin with that of '
                             'the network')
        self._ID = new_ID

    @property
    def network(self):
        """Network associated with the configuration.

        Network associated with the configuration, as a `Network` instance. It
        is set at initialization, but it cannot be reset nor deleted afterwards.
        """
        return self._network

    @property
    def initial_state(self):
        """Initial state of the configuration.

        The initial state of the network. As detailed in the
        [Notes](#configuration-initial-state-notes) section below, if the
        network has *p* populations, the initial state always has *p*(2*p*+3)
        components. When it is set, it is converted to a NumPy array of floats,
        and it is filled with zeros if not enough components are provided. An
        error is raised if it is set to an array with too much components.

        Notes {#configuration-initial-state-notes}
        -----
        For every configuration to be usable with any dynamical system studied
        in the package, the initial state of a configuration should have the
        dimension of the largest of these dynamical systems, which are the
        extended systems. For a network of *p* populations, the extended
        systems have *p*(2*p*+3) dimensions: there are *p* equations for *A*'s,
        *p* equations for *R*'s, *p*(*p*+1)/2 equations for covariances between
        *A*'s, *p*(*p*+1)/2 equations for covariances between *R*'s, and
        *p*<sup>2</sup> equations for covariances between *A*'s and *R*'s. The
        states are ordered as follows:
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
                \\mathrm{C}_{AR}^{p2}, ..., \\mathrm{C}_{AR}^{pp}).
            \\end{aligned}
        \\]
        Remark that there are no \\(\\mathrm{C}_{AA}^{21}\\) or
        \\(\\mathrm{C}_{RR}^{21}\\) components, for example, since the
        \\(\\mathrm{C}_{AA}\\) and \\(\\mathrm{C}_{RR}\\) matrices are symmetric
        and each independant state variable is given only once. This is not the
        case for \\(\\mathrm{C}_{AR}\\), since
        \\[
            \\mathrm{C}_{AR}^{JK} = \\mathrm{Cov}[A_J, R_K] \\neq 
            \\mathrm{Cov}[A_K, R_J] = \\mathrm{C}_{AR}^{KJ} 
        \\]
        in general. 
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, new_state):
        self._initial_state = self._adapt_state(new_state)

    @property
    def Q(self):
        """Input in the network.

        Input in the populations of the network from an external source. It
        must have the same length as the number of populations of the network.
        When it is set, it is automatically converted to a NumPy array of
        floats, and an error is raised if it does not have the correct length.
        """
        return self._Q

    @Q.setter
    def Q(self, new_Q):
        try:
            float_new_Q = float(new_Q)
        except Exception:
            pass
        else:
            new_Q = [float_new_Q]
        if len(new_Q) != (p := len(self.network.populations)):
            raise PopNetError(f'The input must have {p} components for a '
                              f'network of {p} populations.')
        self._Q = np.array(new_Q, float)

    @property
    def initial_time(self):
        """Time from which the network's state is studied.

        Start of the period in which the evolution of the network's state is
        studied. When setting the initial time, the time step
        `Configuration.delta` is adapted to ensure that it is still consistent
        with the number of iterations and the total time interval.
        """
        return self._initial_time

    @initial_time.setter
    def initial_time(self, new_initial_time):
        self._initial_time = float(new_initial_time)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    @property
    def final_time(self):
        """Time until which the network's state is studied.

        End of the period in which the evolution of the network's state is
        studied. When setting the final time, the time step
        `Configuration.delta` is adapted to ensure that it is still consistent
        with the number of iterations and the total time interval.
        """
        return self._final_time

    @final_time.setter
    def final_time(self, new_final_time):
        self._final_time = float(new_final_time)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    @property
    def delta(self):
        """Time step between two iterations.
        
        Time step between two consecutive iterations in numerical experiments.
        It is the time step used in numerical integrations. It is also
        the time step between two points when statistics are computed from
        simulations of the network's microscopic dynamics. This is not used
        when individual simulations of the microscopic dynamics are performed.

        When setting the time step, the number of iterations
        `Configuration.iterations` is adapted to ensure that it is still
        consistent with the time step and the total time interval.
        """
        return self._delta

    @delta.setter
    def delta(self, new_delta):
        self._delta = float(new_delta)
        self._iterations = round((self.final_time - self.initial_time) 
                                 / self.delta)

    @property
    def iterations(self):
        """Number of iterations in the time interval.

        Total number of iterations in the time interval in numerical
        experiments. It is the number of iterations in numerical integrations.
        It is also the number of time steps added after the initial time to get
        the times array when computing statistics from simulations of the
        network's microscopic dynamics. In both cases, the length of the times
        array is `1 + iterations`. This is not used when individual simulations
        of the microscopic dynamics are performed.

        When setting the number of iterations, the time step
        `Configuration.delta` is adapted to ensure that it is still consistent
        with the number of iterations and the total time interval.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, new_number_of_iterations):
        self._iterations = int(new_number_of_iterations)
        self._delta = (self.final_time - self.initial_time) / self.iterations

    def add_random_uniform_perturbation(self, R, axes=None):
        """Add a random perturbation to the initial state.

        Add a random perturbation to the initial state, taken from a uniform
        distribution on an *N*--sphere of radius `R`. The dimension *N* is the
        number of components perturbated as given by `axes`. 

        Parameters
        ----------
        R : float
            Norm of the perturbation. Corresponds to the radius of the
            *N*--sphere in which the perturbation is randomly taken.
        axes : list or tuple of ints, optional
            Axes to change in the initial state. Defaults to `None`, in which
            case every component is changed. 

        Notes
        -----
        To generate a uniform distribution on an *N*--sphere of radius `R`, we
        use the method described in [1]: every component of the perturbation is
        first taken from a standard normal distribution, and then the resulting
        vector is scaled to have a norm of `R`.

        References
        ----------
         1. Muller, M. E. “A note on a method for generating points uniformly on
            *N*-dimensional spheres.” *Commun. ACM* **2**, 19--20 (1959).
            doi:[10.1145/377939.377946](https://doi.org/10.1145/377939.377946).
        """
        if axes is None:
            axes = np.arange(len(self.initial_state))
        ball = np.random.default_rng().normal(size=len(axes))
        ball = R * ball / np.linalg.norm(ball)
        perturbation = np.zeros(len(self.initial_state))
        perturbation[np.array(axes)] = ball
        self.initial_state += perturbation

    def add_to_initial_state(self, perturbation):
        """Add a perturbation to the initial state.
        
        Add a given perturbation to the initial state or to its first
        components.

        Parameters
        ----------
        perturbation : array_like
            Perturbation to add to the initial state. If it is shorter than the
            initial state, it is filled with zeros and then added to the
            initial state.
        """
        if perturbation is None:
            return
        perturbation = self._adapt_state(perturbation)
        self.initial_state += perturbation

    def copy(self, new_ID):
        """Copy the configuration.

        Return a copy of the configuration with a new ID.

        Parameters
        ----------
        new_ID : str
            ID to give to the new configuration.

        Returns
        -------
        Configuration
            The copied configuration.
        """
        new_config = deepcopy(self)
        new_config.ID = new_ID
        return new_config

    def microscopized(self, sizes, new_ID=None):
        """Get a microscopic version of the configuration.

        Return a copy of the configuration, but with a microscopic structure
        where population sizes are given by `sizes`.

        Parameters
        ----------
        sizes : int or list or tuple of int
            Sizes to give to the populations of the network, in the same order
            as in the network's attribute.
        new_ID : str
            ID to give to the new configuration. Defaults to `None`, in which
            case the configuration's ID is used.
        """
        if new_ID is None:
            new_ID = self.ID
        net = self.network.copy(self.network.ID)
        sizes = self._check_list_of_sizes(sizes)
        for pop, size in zip(net.populations, sizes):
            pop.size = size
        try:
            net = net.underlying()
        except AttributeError:
            pass
        configuration = config(net, ID=new_ID)
        for param, value in self.__dict__.items():
            if param in ('_network', '_ID'):
                continue
            setattr(configuration, param, value)
        configuration.reset_micro_initial_state()
        return configuration

    def save(self, folder=None, save_network=True, note=None):
        """Save the current configuration.

        Save the string representation of the configuration in a text file,
        under the name *ID - Configuration.txt*, where *ID* is the actual ID of
        the configuration.

        Parameters
        ----------
        folder : str, optional
            A folder in which the file is saved. If it does not exist in the
            current directory, it is created. Defaults to `None`, in which case
            the file is saved in the current directory.
        save_network : bool, optional
            Decides if the network parameters are saved as well in the same
            folder, using `Network.save`. Defaults to `True`.
        note : str, optional
            If given, an additional section "Additional notes:" is written
            in the file, and `note` is written there. Defaults to `None`, in
            which case this section is not added.
        """
        if save_network:
            self.network.save(folder=folder)
        filename = _internals._format_filename(folder, self.ID, 'Configuration')
        _internals._make_sure_folder_exists(folder)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(self))
            if note is not None:
                file.write('\n\nAdditional notes:\n')
                file.write(note)

    def set_covariances_from_expectations(self, distinct_states):
        """Set initial covariances from initial expectations.

        Set initial values of covariances from initial values of expectations,
        assuming that in the initial state, the states of all neurons are
        taken from ternary distributions on \\(\\{0, 1, i\\}\\) where the
        probability of being in a given state is the expectation of the
        associated fraction of population. Assuming that each population is
        split into *k* subpopulations of the same size and that the neurons of
        a subpopulation are given the same state independently of the states of
        the other subpopulations, then the covariances are determined by the
        number of subpopulations in each population. The precise formula giving
        covariances is provided in the [Notes](#notes) section below.

        Parameters
        ----------
        distinct_states : int or list or tuple of int
            A number of distinct states to draw for each population, given in
            the order prescribed by the network's list of populations.

        Notes
        -----
        We start by the case where each neuron of a population is its own
        subpopulation, so that the states of all neurons are independent.

        Here, covariances are computed assuming that in the initial state, the
        state of each neuron follows a ternary distribution on
        \\(\\{0, 1, i\\}\\) where the probability of being in a given state is
        the expectation of the associated fraction of population. Hence, for
        instance, the probability that a neuron of population *J* is active is
        given by the expected active fraction of *J*. Assuming that the states
        of all neurons are independent, this means that
        \\[
        \\mathrm{C}_{AA}^{JJ}(0) = \\frac{1}{|J|^2} \\sum_{j\\in J}
            \\mathrm{Var}\\bigl[ \\mathrm{Re} X_0^j \\bigr]
            = \\frac{1}{|J|^2} \\sum_{j\\in J}
            \\mathcal{A}_J(0) \\bigl( 1 - \\mathcal{A}_J(0) \\bigr)
            = \\frac{\\mathcal{A}_J(0) (1 - \\mathcal{A}_J(0))}{|J|}.
        \\]
        The same pattern works for the other two states.

        As the states of different neurons are independent, the covariances of
        fractions of distinct populations are zero. However, the covariance
        \\(\\mathrm{C}_{AR}^{JJ}(0)\\) is also determined by the initial
        expectations, as it is determined by variances. Indeed, since the sum
        of all three fractions of the same population is 1 at all times,
        \\[
        \\mathrm{C}_{AR}^{JJ}(0) = \\frac{1}{2} \\bigl(
            \\mathrm{C}_{SS}^{JJ}(0)
                - \\mathrm{C}_{AA}^{JJ}(0)
                - \\mathrm{C}_{RR}^{JJ}(0)
        \\bigr) = - \\frac{\\mathcal{A}_J(0)\\mathcal{R}_J(0)}{|J|}.
        \\]

        In general, a population *J* is split into *N* subpopulations where *N*
        divides |*J*|. Then a subpopulation *k* is given a state \\(Z_0^k\\),
        and each neuron *j* of this subpopulation is given this state as well.
        Therefore, the random variables \\(A_0^J\\) and \\(R_0^J\\) are the
        real and imaginary parts of the random variable
        \\[
        \\frac{1}{|J|} \\sum_{j\\in J} X_0^j
            = \\frac{1}{|J|} \\sum_{k=1}^N \\frac{|J|}{N} Z_0^k
            = \\frac{1}{N} \\sum_{k=1}^N Z_0^k.
        \\]
        Taking the subpopulation states \\(Z_0^k\\) independently, this means
        that the state of *J* has the same distribution as the state of a
        population of *N* neurons with independent states. Thus, all
        covariances can be obtained from the method described above, replacing
        the size of *J* with its number *N* of distinct states.
        """
        distinct_states = self._check_list_of_sizes(distinct_states,
                                                    name='distinct states')
        p = len(self.network.populations)
        VA = np.zeros(len(distinct_states))
        VR = np.zeros(len(distinct_states))
        CovAR = np.zeros(len(distinct_states))
        for J, nJ in enumerate(distinct_states):
            VA[J] = self.initial_state[J]*(1 - self.initial_state[J]) / nJ
            VR[J] = self.initial_state[p+J]*(1 - self.initial_state[p+J]) / nJ
            CovAR[J] = - self.initial_state[J] * self.initial_state[p+J] / nJ
        CAA = np.diag(VA)
        CRR = np.diag(VR)
        CAR = np.diag(CovAR)
        covariances = np.zeros(p*(2*p + 1))
        covariances[: round(p*(p+1)/2)] = CAA[np.triu_indices(p)]
        covariances[round(p*(p+1)/2) : p*(p+1)] = CRR[np.triu_indices(p)]
        covariances[p*(p+1) :] = CAR.flatten()
        self.initial_state[2*p :] = covariances

    def set_initial_state_from(self, other):
        """Set the initial state from another configuration.

        Set the initial state from another configuration, where the network can
        have another number of populations.

         - If the other configuration has *less* populations, only the state
           components associated with the first populations is set.
         - If the other configuration has the *same* number of populations,
           the initial state is simply copied.
         - If the other configuration has *more* populations, the initial
           state is set according to the first populations of the other
           configuration.

        Parameters
        ----------
        other : Configuration
            Other configuration from which to take the initial state.

        Raises
        ------
        TypeError
            If `other` is not a `Configuration` instance.
        """
        if not isinstance(other, Configuration):
            raise TypeError('The initial state can only be copied from another '
                            '\'Configuration\' instance.')
        sp = len(self.network.populations)
        op = len(other.network.populations)
        if op == sp:
            self.initial_state = other.initial_state
            return
        mp = min(sp, op)
        self.initial_state[:mp] = other.initial_state[:mp]
        self.initial_state[sp:sp+mp] = other.initial_state[op:op+mp]
        def loop(start, step):
            sn, on = start(sp), start(op)
            for j in range(mp):
                self.initial_state[sn : sn + step(mp,j)] = \
                    other.initial_state[on : on + step(mp,j)]
                sn, on = sn + step(sp, j), on + step(op, j)
        loop(lambda p: 2*p, lambda p,j: p - j)                    # CAA
        loop(lambda p: 2*p + round(p*(p+1)/2), lambda p,j: p - j) # CRR
        loop(lambda p: 2*p + p*(p+1), lambda p,j: p)              # CAR

    def set_random_initial_state(self, bound_cov=0.06):
        """Set the initial state with random values.

        Set the initial state with random values. For each population, the
        values for *A* and *R* are chosen from uniform distributions in the
        triangle \\(\\{(x,y) \\in [0,1)^2 : x + y < 1\\}\\), using the method
        described in [1]. All variances are chosen from uniform distributions
        between 0 and `bound_cov`, and all non-symmetric covariances from
        uniform distributions between `-bound_cov` and `bound_cov`, regardless
        of the values of the expectations. 

        Parameters
        ----------
        bound_cov : float, optional
            Positive number that sets the distributions of covariances.
            Variances are all taken from a uniform distribution between zero and
            `bound_cov`, and non-symmetric covariances are taken from a uniform
            distribution between `-bound_cov` and `bound_cov`. Defaults to 0.06.

        Raises
        ------
        TypeError
            If `bound_cov` cannot be converted to a float.
        ValueError
            If `bound_cov` is not a positive number.

        References
        ----------
         1. Osada, R., Funkhouser, T., Chazelle, B. & Dobkin, D. “Shape
            distributions.” *ACM Trans. Graph.* **21**, 807--832 (2002).
            doi:[10.1145/571647.571648](https://doi.org/10.1145/571647.571648).
        """
        p = len(self.network.populations)
        state = np.zeros(p*(2*p+3))
        rng = np.random.default_rng()
        try:
            bound_cov = float(bound_cov)
        except TypeError as error:
            raise TypeError('The bound to choose covariances must be a number.'
                            'number.') from error
        if bound_cov < 0:
            raise ValueError('The bound to choose covariances must be '
                             'positive.')
        for J in range(p):
            a, b = rng.random(size=2)
            state[J] = np.sqrt(a) * (1 - b)
            state[p+J] = np.sqrt(a) * b
        state[2*p : p*(p+3)] = bound_cov * rng.random(size=p*(p+1))
        state[p*(p+3) :] = -bound_cov + 2*bound_cov * rng.random(size=p**2)
        self.initial_state = state

    def _adapt_state(self, state):
        """Check a state and fill it with zeros if necessary."""
        try:
            state = np.array(state, float, ndmin=1)
        except ValueError as error:
            raise ValueError('Can\'t convert the state to an array of floats.')
        p = len(self.network.populations)
        diff = p * (2*p + 3) - len(state)
        if diff > 0:
            state = np.concatenate((state, np.zeros(diff)))
        elif diff < 0:
            raise ValueError(f'A state cannot have more than {p*(2*p+3)} '
                             f'components for a network of {p} populations.')
        return state

    def _check_list_of_sizes(self, sizes, name='sizes'):
        """Check a list of population sizes or similar."""
        if isinstance(sizes, int):
            sizes = [sizes]
        try:
            sizes = list(sizes)
        except TypeError as error:
            raise TypeError(f'The given {name} could not be converted to a '
                            'list.') from error
        if len(sizes) != len(self.network.populations):
            raise ValueError(f'The list of {name} must have the same length as '
                            'the list of populations of the network.')
        return sizes

    def _other_params_string(self):
        """Optional parameters for subclasses to write in `str(self)`."""
        pass


class MicroConfiguration(Configuration):
    """Configurations used in numerical stochastic simulations.

    `MicroConfiguration` extends `Configuration` to cases where the microscopic
    structure of the network is needed to perform numerical simulations. It
    adds two new properties:

     - `MicroConfiguration.micro_initial_state`, which gives the network's
       microscopic initial state;
     - `MicroConfiguration.executions`, which gives the number of trajectories
       generated when performing simulations in chain.

    It also provides a method to reset the microscopic initial state from the
    macroscopic one, and another to modify the sizes of the populations of the
    network while also updating the network's parameters and the microscopic
    initial state.

    Since this configuration class requires the network to have a microscopic
    structure, it has to be initialized from a `MicroNetwork` instance. Besides
    that, the initialization is the same as in the base class.

    Raises
    ------
    popnet.exceptions.PopNetError
        If the network used is not a `MicroNetwork`.

    """

    def __init__(self, network, ID=None, **kwargs):
        if 'executions' in kwargs:
            self.executions = kwargs.pop('executions')
        else:
            self.executions = 1
        super().__init__(network, ID=ID, **kwargs)
        if not isinstance(network, MicroNetwork):
            raise PopNetError('The network used with a \'MicroConfiguration\' '
                              'must be a \'MicroNetwork\'.')

    @Configuration.initial_state.setter
    def initial_state(self, new_state):
        Configuration.initial_state.fset(self, new_state)
        self.reset_micro_initial_state()

    @property
    def micro_initial_state(self):
        """Microscopic initial state of the network.

        States of all neurons of the network. It is always consistent with the
        macrosopic initial state `Configuration.initial_state`, in the sense
        that the microscopic initial state can only be set from macroscopic one,
        and that it is automatically reset when the macroscopic initial state
        is set. For more details on this process, see
        `MicroConfiguration.reset_micro_initial_state`.

        The microscopic initial state cannot be set manually, but it can be
        reset at any time with `MicroConfiguration.reset_micro_initial_state`.
        Note also that when the macroscopic initial state is changed, the
        microscopic one is also reset.
        """
        return self._micro_initial_state

    @property
    def executions(self):
        """Number of simulations to be done.

        Number of simulations to be performed when doing simulations in chain
        to obtain statistics. It cannot be deleted.
        """
        return self._executions

    @executions.setter
    def executions(self, new_number):
        try:
            new_number = int(new_number)
        except TypeError as error:
            raise TypeError('The number of simulations to be done must be '
                            'a number.') from error
        if new_number < 0:
            raise ValueError('The number of simulations to be done must be '
                             'positive.')
        self._executions = new_number

    def resize_network(self, new_sizes):
        """Change the sizes of the network's populations.

        Change the size of each population of the network, and reset the
        network's parameters and the microscopic initial state to be consistent
        with this change.

        Parameters
        ----------
        new_sizes : int or list or tuple of int
            A new size for each population, given in the order prescribed by
            the network's list of populations.
        """
        new_sizes = self._check_list_of_sizes(new_sizes)
        for pop, new_size in zip(self.network.populations, new_sizes):
            pop.size = new_size
        self.network.reset_parameters()
        self.reset_micro_initial_state()
        
    def reset_micro_initial_state(self, distinct_states=None):
        """Randomly generate a microscopic initial state.
        
        Create for the network a microscopic initial state consistent with its
        macroscopic initial state. By default, if *J* is a population of the
        network, the state of each neuron of *J* is chosen randomly between the
        values `1` (active), `1j` (refractory) and `0` (sensitive), with
        probabilities corresponding to the active, refractory and sensitive
        fractions of *J*. The parameter `distinct_states` can be used to force
        subpopulation of neurons to have the exact same state: the whole
        population will be split into as much subpopulations as distinct states
        are drawn, and then all neurons within a subpopulation will be given
        the same state.

        Parameters
        ----------
        distinct_states : int or list or tuple of int, optional
            A number of distinct states for each population, given in the order
            prescribed by the network's list of populations. It is assumed that
            the number of distinct states for each population divides its size.
            Defaults to `None`, in which case it is replaced by the list of
            population sizes and each neuron will be given a distinct state.

        Raises
        ------
        ValueError
            If the provided numbers of distinct states do not divide the
            population sizes.
        """
        if distinct_states is None:
            distinct_states = [pop.size for pop in self.network.populations]
        else:
            distinct_states = self._check_list_of_sizes(distinct_states,
                                                        name='distinct states')
            for J, popJ in enumerate(self.network.populations):
                distinct_states[J] = int(distinct_states[J])
                if popJ.size % distinct_states[J] != 0:
                    raise ValueError('The number of distinct states taken for '
                                     f'population {J+1} should divide its size')
        A = self.initial_state[: (p := len(self.network.populations))]
        R = self.initial_state[p : 2*p]
        S = 1 - A - R
        rng = np.random.default_rng()
        generated_micro_state = np.concatenate(
            [rng.choice((0., 1., 1j), p=(S[J], A[J], R[J]), size=n_states)
             for J, n_states in enumerate(distinct_states)])
        repetitions = np.concatenate(
            [[pop.size // n_states for j in range(n_states)]
             for pop, n_states in zip(self.network.populations, distinct_states)])
        self._micro_initial_state = np.concatenate(
            [[state for j in range(rep)]
             for state, rep in zip(generated_micro_state, repetitions)])

    def set_covariances_from_expectations(self, distinct_states=None):
        """Set initial covariances from initial expectations.
        
        Extends the base class method by taking, by default, the populations'
        sizes as the numbers of distinct states to define covariances.

        Parameters
        ----------
        distinct_states : int or list or tuple of int, optional
            A number of distinct states to draw for each population, given in
            the order prescribed by the network's list of populations. Defaults
            to `None`, in which case the populations' sizes are used.
        """
        if distinct_states is None:
            distinct_states = [pop.size for pop in self.network.populations]
        super().set_covariances_from_expectations(distinct_states)

    def _other_params_string(self):
        """Add `executions` to `str(self)`."""
        if self.executions == 1:
            return f'{1:>9} execution\n'
        else:
            return f'{self.executions:>9} executions\n'


class ConfigurationOne(Configuration):
    """Extends `Configuration` in the special case of a single population.

    Extends `Configuration` by adding methods specific to the case of only one
    population. The new methods allow to:

     - Verify if a state is in the domain where variables make sense,
       physiologically speaking;
     - Set the initial state randomly in this physiological domain;
     - Set the input and the initial state to coordinates where there is a fixed
       point.

    The initialization is the same as in the base class, except for a
    verification that the network has indeed a single population.

    The data attributes are the same as in the base class.

    Raises
    ------
    popnet.exceptions.PopNetError
        If the network does not have precisely one population.

    """

    def __init__(self, network, ID=None, **kwargs):
        super().__init__(network, ID=ID, **kwargs)
        if network.ID[0] != '1':
            raise PopNetError('\'ConfigurationOne\' can be used only in cases '
                              'where the network has a single population. The '
                              f'network used here has {network.ID[0]}')
        self._variables = ['A', 'R', 'CAA', 'CRR', 'CAR']

    def set_random_initial_state(self, domain='physiological', bound_cov=0.06):
        """Set the initial state randomly.

        Overrides the corresponding base class method to choose an initial state
        in a given domain. *A* and *R* are always chosen from a uniform
        distribution in the triangle \\(\\{(x,y) \\in [0,1)^2 : x + y < 1\\}\\).

        Parameters
        ----------
        domain : {'physiological', 'bounded'}, optional
            The domain in which the state is chosen. If 'physiological', the
            state is chosen in the so-called physiological domain, where
            expectations and covariances are valid values for the random
            variables they represent. If 'bounded', the base class method is
            called. Defaults to `'physiological'`.
        bound_cov : float, optional
            Positive number which sets the distributions of covariances, in the
            case where `domain` is set to `'bounded'`. See the base class method
            for details. Defaults to 0.06.

        Raises
        ------
        NotImplementedError
            If the requested domain is not valid. 

        See Also
        --------
        Configuration.set_random_initial_state
        """
        rng = np.random.default_rng()
        if domain == 'physiological':
            a, b = rng.random(size=2)
            A = np.sqrt(a) * (1 - b)
            R = np.sqrt(a) * b
            CAA = A * (1 - A) * rng.random()
            CRR = R * (1 - R) * rng.random()
            CAR = -np.sqrt(CAA*CRR) + 2*np.sqrt(CAA*CRR) * rng.random()
            new_state = [A, R, CAA, CRR, CAR]
            if self.state_in_domain(new_state, verbose=False):
                self.initial_state = new_state
            else:
                self.set_random_initial_state()
            return
        elif domain == 'bounded':
            super().set_random_initial_state(bound_cov=bound_cov)
            return
        raise NotImplementedError(f'No domain \'{domain}\' available.')

    def set_to_fixed_point(self, form, set_state=True):
        """Set the input and initial state to a fixed point.

        Set the input and (possibly) the initial state to coordinates where
        there is a fixed point of form *i*), *ii*) or *iii*) of the extended
        Wilson--Cowan system obtained from the closure that uses a second-order
        Taylor approximation.

        Parameters
        ----------
        form : {'i', 'ii', 'iii'}
            The desired form of fixed point.
        set_state : bool, optional
            Decides if the initial state is set to the fixed point. Defaults
            to `True`.

        Raises
        ------
        NotImplementedError
            If `form` is `'i'`. 
        ValueError
            If `form` is neither of `'i'`, `'ii'` or `'iii'`.
        """
        pop = self.network.populations[0]
        c = self.network.c[0,0]
        Sigma = pop.alpha/2 + pop.beta + pop.gamma
        Pi = pop.alpha*pop.beta/2 + pop.alpha*pop.gamma/2 + pop.beta*pop.gamma
        if form == 'i':
            raise NotImplementedError('The coordinates for a form i) fixed '
                                      'point have not been implemented yet.')
        elif form == 'ii':
            S = 4 * pop.scale_theta * Sigma / (pop.alpha * c)
            A = pop.gamma / (pop.beta + pop.gamma) * (1 - S)
            R = pop.beta  / (pop.beta + pop.gamma) * (1 - S)
            CAR = (2*pop.gamma / (pop.beta+pop.gamma) * (Sigma * Pi - pop.beta
                    * pop.gamma * pop.alpha * c / (4*pop.scale_theta))
                    / ( (pop.alpha + 2*pop.gamma) * 
                        (pop.alpha * c / (4*pop.scale_theta))**2 ))
            CAA = pop.alpha / (2*pop.gamma) * CAR
            CRR = pop.beta / pop.gamma * CAR
            self.Q = pop.theta - c * A
        elif form == 'iii':
            S = 4 * pop.scale_theta * Pi / (pop.gamma * pop.alpha * c)
            A = pop.gamma / (pop.beta + pop.gamma) * (1 - S)
            R = pop.beta  / (pop.beta + pop.gamma) * (1 - S)
            CAR = (pop.beta / pop.gamma * ( Pi**2 - pop.beta*pop.gamma**2
                    * pop.alpha * c / (4*pop.scale_theta) )
                    / ( (pop.beta+pop.gamma)**2 
                        * (pop.alpha * c / (4*pop.scale_theta))**2 ))
            CAA = pop.gamma / pop.beta * CAR
            CRR = pop.beta / pop.gamma * CAR
            self.Q = pop.theta - c * A
        else:
            raise ValueError('Unknown fixed point form.')
        if set_state:
            self.initial_state = [A, R, CAA, CRR, CAR]

    def state_in_domain(self, state=None, verbose=False):
        """Verify if a state is in the physiological domain.

        Parameters
        ----------
        state : array_like, optional
            The state to verify. Defaults to `None`, in which case the initial
            state is verified.
        verbose : bool, optional
            If `True`, a warning is issued if the state is not in the
            physiological domain. Defaults to `False`.

        Returns
        -------
        bool
            `True` if the state is in the physiological domain, else `False`.

        Warns
        -----
        popnet.exceptions.PopNetWarning
            If `verbose` is `True` and if the state is not in the physiological
            domain.
        """
        if state is None:
            state = self.initial_state
        A = state[0]
        R = state[1]
        S = 1 - A - R
        CAA = state[2]
        CRR = state[3]
        CAR = state[4]
        CSS = CAA + 2*CAR + CRR
        if A < 0 or R < 0 or S < 0:
            ok = False
        elif CAA > A * (1 - A) or CAA < 0:
            ok = False
        elif CRR > R * (1 - R) or CRR < 0:
            ok = False
        elif CSS > S * (1 - S) or CSS < 0:
            ok = False
        elif CAR**2 > CAA * CRR:
            ok = False
        elif CAR < -A*R:
            ok = False
        elif CAR > A*S - CAA:
            ok = False
        elif CAR > R*S - CRR:
            ok = False
        else:
            ok = True
        if not ok and verbose:
            warn(f'The state {state} does not make sense, physiologically '
                 'speaking.', category=PopNetWarning, stacklevel=2)
        return ok


class MicroConfigurationOne(ConfigurationOne, MicroConfiguration):
    """Extends `MicroConfiguration` in the special case of a single population.

    Combines the features of `ConfigurationOne` and `MicroConfiguration` in
    order to be used for cases where the microscopic structure of a network
    containing a single population is needed.

    """

    pass


def build_network(ID, matrix):
    """Get a network from a weight matrix.

    Get a network with a weight matrix given by `matrix`.

    Parameters
    ----------
    ID : str
        ID given to the network.
    matrix : array_like
        Weight matrix specifying the connections between neurons in the network.
        It has to be square with real entries.

    Returns
    -------
    MicroNetwork
        Network initialized with weight matrix corresponding to `matrix`.
    """
    N = matrix.shape[0]
    if matrix.shape != (N, N):
        raise ValueError('The given matrix must be square.')
    net = default_network(ID, scale='micro')
    net.c = N * np.mean(matrix)
    net.W = matrix
    return net


def config(network, ID=None, **kwargs):
    """Define a configuration from a given network.

    Define a new configuration using the most appropriate class constructor
    according to the number of populations of the network and to the type of
    network.

    Parameters
    ----------
    network : Network
        Network to use with the configuration.
    ID : str, optional
        ID to associate with the configuration. Defaults to `None`, in which
        case the network's ID is used.
    **kwargs
        Keyword arguments to be passed to the class constructor.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        A configuration initialized with the given ID and network. It is an
        instance of a subclass of `Configuration` if it is more appropriate
        according to `scale` and to the number of populations of the network.
    """
    if ID is None:
        ID = network.ID
    if isinstance(network, MicroNetwork) and ID[0] == '1':
        return MicroConfigurationOne(network, ID=ID, **kwargs)
    if isinstance(network, MicroNetwork):
        return MicroConfiguration(network, ID=ID, **kwargs)
    if ID[0] == '1':
        return ConfigurationOne(network, ID=ID, **kwargs)
    return Configuration(network, ID=ID, **kwargs)


def default_config(ID, scale='macro'):
    """Define a configuration with default parameters.

    Define a new configuration with default parameters and a given ID. The
    network associated with this configuration is defined with
    `default_network`, using the same `scale`.

    Parameters
    ----------
    ID : str
        ID given to the new configuration. Its first character must be a
        positive integer and is taken to be the number of populations.
    scale : {'macro', 'micro'}, optional
        Determines whether the new network has a defined microscopic structure.
        If `'micro'`, a default size of 100 neurons is given to each population.
        If `'macro'`, population sizes remain undefined.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        The configuration with default parameters. It is an instance of a
        subclass of `Configuration` if it is more appropriate according to
        `scale` and to the number of populations of the network.

    Raises
    ------
    popnet.exceptions.PopNetError
        If a non-valid value is passed to `scale`.
    """
    net = default_network(ID, scale=scale)
    return config(net, ID)


def default_network(ID, scale='macro'):
    """Define a network with default parameters.
    
    Define a new network with default parameters and a given ID.

    Parameters
    ----------
    ID : str
        ID given to the new network. Its first character must be a positive
        integer, which is taken to be the number of populations.
    scale : {'macro', 'micro'}, optional
        Determines whether the new network has a defined microscopic structure.
        If `'micro'`, a default size of 100 neurons is given to each population.
        If `'macro'`, population sizes remain undefined.

    Returns
    -------
    Network or MicroNetwork
        The network with default parameters. It is a `MicroNetwork` instance
        if the sizes of the populations are defined.

    Raises
    ------
    popnet.exceptions.PopNetError
        If a non-valid value is passed to `scale`.
    """
    p = int(ID[0])
    if p == 1:
        pops = [Population('Population')]
    else:
        pops = [Population(f'Population {j+1}', ID=str(j+1)) for j in range(p)]
    if scale == 'macro':
        return Network(ID, pops)
    elif scale == 'micro':
        for pop in pops:
            pop.size = 100
        return MicroNetwork(ID, pops)
    raise PopNetError(f'Unknown scale {scale} to create a network.')


def load_config(load_ID, new_ID=None, network=None, folder=None):
    """Load a configuration from a text file.

    Load the configuration parameters from a text file. This text file is
    expected to be named *ID - Configuration.txt* (with *ID* replaced with
    the configuration's actual ID). When reading it as a single string, it
    is also expected to have the format of a string representation of a
    `Configuration` instance. Note that this is the format of text file saved
    by `Configuration.save`.

    Parameters
    ----------
    load_ID : str
        ID of the configuration to load. 
    new_ID : str, optional
        ID of the configuration to create with the parameters of `load_ID`.
        Defaults to `None`, in which case `load_ID` is used. 
    network : Network, optional
        The network to associate with the configuration. Defaults to `None`,
        in which case the configuration's network is created with `load_network`.
        In that case, the network is loaded from the ID given in the
        configuration's file, and the network's file is expected to be located
        in the same folder as the configuration's.
    folder : str, optional
        Folder in which the text file is located. If given, if must be in
        the current directory. Defaults to `None`, in which case the text
        file is expected to be in the current directory.

    Returns
    -------
    Configuration, MicroConfiguration, ConfigurationOne or MicroConfigurationOne
        The loaded configuration with ID `new_ID`. It is an instance of a
        subclass of `Configuration` if it is more appropriate according to the
        number of populations of the network and to their sizes.

    Raises
    ------
    FileNotFoundError
        If no file is found with the expected name.
    popnet.exceptions.PopNetError
        If the file contains inconsistent information or unexpected
        parameters to set.
    popnet.exceptions.FormatError
        If the file does not have the expected format.

    Warns
    -----
    popnet.exceptions.PopNetWarning
        If the information in the file is inconsistent.
    """
    filename = _internals._format_filename(folder, load_ID, 'Configuration')
    if new_ID is None:
        new_ID = load_ID
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError as error:
        raise FileNotFoundError('No file found to load configuration '
                                f'{load_ID}.') from error
    loaded_config = _read_config_file(load_ID, new_ID, network, folder, lines)
    if loaded_config is None:
        raise FormatError(f'The file \'{load_ID} - Configuration.txt\' does not'
                          ' have the expected format to load a configuration.')
    return loaded_config


def load_network(load_ID, new_ID=None, folder=None):
    """Load a network from a text file.

    Define a new network from parameters in a text file. This file is expected
    to be named *ID - Network parameters.txt* where *ID* is the network's
    actual ID. If the file is read as a single string, it is expected to have
    the format of a string representation of a `Network` instance. Note that
    this is the format of text file saved by `Network.save`.

    Parameters
    ----------
    load_ID : str
        ID of the network to load. 
    new_ID : str, optional
        ID of the network to define. Defaults to `None`, in which case
        `load_ID` is used. 
    folder : str, optional
        Folder in which the text file is located. If given, if must be in
        the current directory. Defaults to `None`, in which case the text
        file is expected to be in the current directory.

    Returns
    -------
    Network or MicroNetwork
        The loaded network. It is a `MicroNetwork` instance if a size is
        defined for every population.

    Raises
    ------
    FileNotFoundError
        If no file is found with the expected name.
    popnet.exceptions.PopNetError
        If the information in the file is not consistent with `load_ID`.
    popnet.exceptions.FormatError
        If the file does not have the expected format.

    Warnings
    --------
    While it is intended that network files contain data on all parameters,
    missing population parameters are in fact silently ignored and replaced
    with default values. The same goes for missing connection scales matrices.
    This behavior is retained only for compatibility reasons and is not
    guaranteed to work in all cases.
    """
    filename = _internals._format_filename(folder, load_ID, 'Network parameters')
    if new_ID is None:
        new_ID = load_ID
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError as error:
        raise FileNotFoundError('No file found to load network '
                                f'{load_ID}.') from error
    if load_ID != (other_ID := lines[0].strip().split()[-1]):
        raise PopNetError(f'The file \'{load_ID} - Network parameters.txt\' '
                          'seems to contain information about a network '
                          f'{other_ID} rather than {load_ID}.')
    j = 2
    populations = []
    while j < len(lines):
        if lines[j].startswith('Connection matrix'):
            break
        # At index j+1 a population's description starts. We loop until the
        # end of the description, that is, when an empty line is reached.
        k = j + 1
        while lines[k] != '\n':
            k += 1
        populations.append(Population._load(lines[j : k]))
        j = k + 1
    else:
        raise FormatError(f'The file \'{load_ID} - Network parameters.txt\' do'
                          'es not have the expected format to load a network.')
    if any(pop.size is None for pop in populations):
        net = Network(new_ID, populations)
    else:
        net = MicroNetwork(new_ID, populations)
    # Set the connection matrix from the file's data.
    p = len(net.populations)
    string_c = ''.join(lines[j+1 : j+1+p])
    net.c = _get_matrix_from_string(string_c)
    # If the weights' scales are not written in the file, pass.
    string_scale_c = ''.join(lines[j+3+p : j+3+2*p])
    try:
        scale_c = _get_matrix_from_string(string_scale_c)
    except FormatError:
        raise
    except Exception:
        pass
    else:
        net.scale_c = scale_c
    if isinstance(net, MicroNetwork):
        net.reset_parameters()
    return net


def network(ID, populations):
    """Define a network from given populations.

    Define a network from populations that are already defined, possibly with
    a microscopic structure.

    Parameters
    ----------
    ID : str
        ID given to the network.
    populations : list or tuple of Population or Population
        Populations that define the network.

    Returns
    -------
    Network or MicroNetwork
        The network made of the populations listed in `populations`. It is a
        `MicroNetwork` instance when all populations have a defined size.

    Raises
    ------
    TypeError
        If `populations` cannot be converted into a tuple of `Population`
        instances.
    """
    net = Network(ID, populations)
    try:
        net = net.underlying()
    except PopNetError:
        pass
    return net


def population(name, ID=None, size=None):
    """Define a population of biological neurons.

    Define a `Population` instance with a given name, possibly with a
    microscopic structure.

    Parameters
    ----------
    name : str
        Name of the population.
    ID : str, optional
        ID of the population. It must be a single character. If not given, the
        last character of `name` is used if it is a number, else the first one
        is used.
    size : int, optional
        Number of neurons of the population. If given, it must be positive.
        Defaults to `None`, in which case no size is defined for the population.

    Returns
    -------
    Population
        The population initialized with the given name, ID and size.
    """
    pop = Population(name, ID=ID, size=size)
    return pop


def _get_matrix_from_string(string):
    """Get a network connection matrix from a string.

    Parameters
    ----------
    string : str
        String from which to read the matrix. It should have the format of a
        string representation of a NumPy array.

    Raises
    ------
    FormatError
        If the string does not have the correct format.
    """
    if string[-1] == '\n':
        string = string[:-1]
    string = re.sub(r'\[\s+', '[', string)
    string = re.sub(r'\s+\]', ']', string)
    string = re.sub(r'\s+', ',', string)
    try:
        matrix = ast.literal_eval(string)
    except Exception as err:
        raise FormatError('The string could be converted to a matrix.') from err
    return matrix


def _read_config_file(load_ID, new_ID, network, folder, lines):
    """Read a configuration file given as a list of lines.

    If the file does not have the expected format, return `None`. Else, return
    loaded configuration. See `popnet.load_config` for details.
    """
    if not lines[0].startswith('Configuration'):
        return None
    if load_ID != (other_ID := lines[0].strip().split()[-1]):
        raise PopNetError(f'The file \'{load_ID} - Configuration.txt\' seems '
                          'to contain information about a configuration '
                          f'{other_ID} rather than {load_ID}.')
    if not lines[2].startswith('Network'):
        return None
    # If no network was given, load it from the ID given in the file.
    if network is None:
        network_ID = lines[2].strip().split()[-1]
        network = load_network(network_ID, folder=folder)
    # Define the configuration.
    configuration = config(network, ID=new_ID)
    if not lines[4].startswith('Parameters'):
        return None
    # Load parameters. 
    params = {'initial_time': None, 'final_time': None, 
              'iterations': None, 'delta': None, 'executions': None}
    for j in range(5, len(lines)):
        if lines[j].startswith('Input'):
            break
        words = lines[j].strip().split()
        if len(words) == 0:
            continue
        elif len(words) == 2:
            param = words[1]
            val = words[0]
        elif len(words) == 3:
            param = words[0].replace('ti', 'initial_time')
            param = param.replace('tf', 'final_time')
            param = param.replace(u'\u0394t', 'delta')
            val = words[2]
        else:
            raise FormatError(f'Unexpected line {lines[j]} in saved file for '
                              f'configuration {load_ID}.')
        if param in params:
            params[param] = val
        elif param == 'execution':
            params['executions'] = val
        else:
            raise PopNetError(f'Unexpected parameter {param} to load with '
                              f'configuration {load_ID}')
    else:
        # If the loop ended without breaking, something went wrong.
        return None
    for param in params:
        if param == 'delta' and params['iterations'] is not None:
            if params[param] not in (None, str(configuration.delta)):
                warn(f'The time interval given in the file {load_ID} - Configu'
                     f'ration.txt is {params[param]}. However, with the initial'
                     f' time of {configuration.initial_time}, the final time of'
                     f' {configuration.final_time}, and '
                     f'{configuration.iterations} iterations, the time interval'
                     f' should have been {configuration.delta}. The value given'
                     ' in the file has been replaced by the latter.', 
                     category=PopNetWarning, stacklevel=3)
                continue
        if params[param] is not None:
            setattr(configuration, param, params[param])
    # Set the input value.
    string_Q = lines[j+1].split('=')[-1].strip()
    configuration.Q = ast.literal_eval(re.sub(r'\s+', ',', string_Q))
    if not lines[j+3].startswith('Initial state'):
        return None
    # Set the initial state.
    state = []
    for line in lines[j+4 : j+4+len(configuration.initial_state)]:
        state.append(float(line.strip().split()[-1]))
    configuration.initial_state = state
    return configuration

"""Functions to run numerical experiments.

This module defines several classes dedicated to run numerical experiments using
the data structures defined in `popnet.structures` and the dynamical systems
defined in `popnet.systems`. Its methods allow both to

 - Perform easily the numerical integration of several dynamical systems related
   to the Wilson--Cowan model;
 - Perform simulations to study sample trajectories of a stochastic process
   which is macroscopically approximated by the Wilson--Cowan model.

The main classes defined in the module are briefly described in the
[Classes And Hierarchy](#classes-and-hierarchy) section below. 

Classes and hierarchy
---------------------
The important classes of the module are summarized below. The indentation
follows the hierarchy. 

 - `Executor` : Abstract base class giving an interface to run numerical
   experiments.
     - `Integrator` : An interface to run numerical integrations.
     - `Simulator` : An interface to run simulations of a stochastic process.
         - `SimpleSimulator` : A simulator to run simulations one at a time.
         - `ChainSimulator` : A simulator to run multiple simulations at once.

"""

import numpy as np
from scipy.integrate import ode
from tqdm import tqdm
from warnings import warn
from functools import singledispatch

from .exceptions import *
from . import _internals
from . import structures
from . import systems
from . import graphics


class Executor:
    """Execute numerical experiments on a network.

    `Executor` is meant to perform numerical experiments to study the dynamics
    of a network split into populations. These experiments are intended to be
    carried out by subclasses of `Executor`.

     - To perform simulations of the stochastic process which rules the
       evolution of the network, use `Simulator`.
     - To perform numerical integrations of reduced dynamical systems
       describing the macroscopic behavior of the network, use `Integrator`.

    A configuration must be given, and a reset of the executor is made at the
    end of the initialization, when setting the configuration.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration used for the experiments.

    Attributes
    ----------
    config : popnet.structures.Configuration
        Configuration used for the experiments. See `Executor.config`.
    times : array_like
        Time. See `Executor.times`.
    states : array_like
        State of the network with respect to time. See `Executor.states`.

    """

    def __init__(self, config):
        self.config = config
        self._output_type = None

    @property
    def config(self):
        """Configuration used with the executor.

        Configuration defining all parameters used by the executor. It must be
        a `popnet.structures.Configuration` instance, or a
        `popnet.structures.MicroConfiguration` instance if the network should
        have a microscopic structure. If it is set, the executor is reset with
        `Executor.reset`. It cannot be deleted.
        """
        return self._config

    @config.setter
    def config(self, new_value):
        if not isinstance(new_value, structures.Configuration):
            raise TypeError('The configuration used with an executor must be a '
                            '\'Configuration\' instance.')
        self._config = new_value
        self.reset()

    @property
    def states(self):
        """State of the network with respect to time.

        Macroscopic state of the network at each time step. It does not contain
        any relevant data at initialization or right after a reset, but it is
        updated during a call to `Executor.run`. It cannot be manually set nor
        deleted.
        """
        return self._states

    @property
    def times(self):
        """Time.

        At initialization or with a call to `Executor.reset`, it is set
        according to the integrator's configuration `config`. Specifically, it
        is an array starting at `config.initial_time` and ending at
        `config.final_time`, with an interval of `config.delta` between time
        steps. It cannot be manually set nor deleted.
        """
        return self._times

    @property
    def success(self):
        """Indicator of success of a numerical experiment.

        Indicator of the success of a numerical experiment. It is set to `None`
        when the executor is reset, and then set to a boolean value after an
        experiment has been performed to indicate whether it was successful or
        not. It cannot be manually set nor deleted.
        """
        return self._success

    def close(self):
        """Delete all data attributes of the executor."""
        del self._config
        del self._states
        del self._times
        del self._success

    def output(self, **kwargs):
        """Get the output of the execution.

        Return the results of the numerical experiment.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the output's class constructor.

        Returns
        -------
        Result
            The output of the experiment. The precise output type depends on the
            experiment executed and on the number of populations of the network.
            See the [summary](graphics.html#classes-and-hierarchy) of all
            `popnet.graphics.Result` subclasses for a quick reference giving
            the output type of each case.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the numerical experiment has not been performed yet.
        """
        self._check_if_run()
        if self._output_type is None:
            raise PopNetError(
                'PopNet does not know how to output the results of this '
                'experiment. It might be due to the use of the base class '
                '\'Result\' rather than its subclasses, or to the numerical '
                'integration of an unrecognized dynamical system. It should '
                'still be possible to save the results with \'save_output\'.')
        return self._output_type(self.config, self._output_states(), 
                                 self._output_times(), **kwargs)

    def reset(self):
        """Reset the executor."""
        self._success = None
        self._times = np.linspace(self.config.initial_time, 
                                  self.config.final_time, 
                                  1 + self.config.iterations)

    def run(self):
        """Run the numerical experiment.
        
        This method is abstract and is implemented in subclasses.
        """
        raise NotImplementedError('An Executor must implement a \'run\' '
                                  'method.')

    def save_output(self, name=None, folder=None):
        """Save the output of the experiment in a text file.

        Save the output of the numerical experiment in a text file, under the
        name *ID - name.txt*, where *ID* is the ID of the configuration used
        for the experiment and *name* is `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved output. Defaults to `None`, in which case
            it is replaced with a default based on the output class.
        folder : str, optional
            A folder in which the file is saved. If it does not exist in the
            current directory, it is created. Defaults to `None`, in which case
            the file is saved in the current directory.

        Returns
        -------
        name : str
            The name given to the saved output.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the numerical experiment has not been performed yet.
        """
        self._check_if_run()
        try:
            name = self._output_type._get_name(name)
        except AttributeError:
            name = 'Output'
        filename = _internals._format_filename(folder, self.config.ID, name)
        _internals._make_sure_folder_exists(folder)
        L = self._state_length()
        header = ''.join([f'{X:<16}' for X in self.config._variables[:L]])
        np.savetxt(filename, self._output_states(), fmt='%+.12f', header=header)
        return name

    def _check_if_run(self):
        """Check if the executor has already run."""
        if self.success is None:
            raise PopNetError('An executor has to run before the results are '
                              'output. Call run() first.')

    def _output_states(self):
        """States array to output."""
        return self.states

    def _output_times(self):
        """Times array to output."""
        return self.times

    def _state_length(self):
        """Length of the macroscopic states."""
        raise NotImplementedError('An executor must give its states\' sizes.')


class Integrator(Executor):
    """Numerical integrator for ODEs related to Wilson--Cowan's model.

    `Integrator` extends `Executor` to perform numerical integrations of
    dynamical systems related to the Wilson--Cowan model. All numerical
    integrations are performed with the class
    [ode](https://tinyurl.com/scipy-integrate-ode) from SciPy's `integrate`
    module. Specific vector fields are implemented in `Integrator`'s subclasses.

    Parameters
    ----------
    system : popnet.systems.DynamicalSystem
        Sets the dynamical system to be integrated.

    Attributes
    ----------
    config : popnet.structures.Configuration
        Configuration used for the numerical integration. It is taken as that
        of `Integrator.system`. See `Integrator.config`.
    system : popnet.systems.DynamicalSystem
        Dynamical system used for the integration. See `Integrator.system`.
    states, times : array_like
        Arrays representing the state of the network with respect to time. See
        `Integrator.states` and `Integrator.times`.

    """

    def __init__(self, system, **kwargs):
        self.system = system
        super().__init__(system.config, **kwargs)
        try:
            output_type = OUTPUT_TYPES[type(self.system)]
        except KeyError:
            warn('The type of dynamical system you want to integrate is not '
                 'recognized by PopNet. The integration should be possible, '
                 'but it will not be possible to output the results with '
                 '\'Integrator.output\'.', category=PopNetWarning, stacklevel=2)
        else:
            self._output_type = output_type

    @property
    def system(self):
        """Dynamical system used for the integration.

        The dynamical system used when performing numerical integrations. It
        must be a `popnet.systems.DynamicalSystem` instance, and it is expected
        to be associated with the same configuration as the integrator. It
        cannot be deleted.
        """
        return self._system

    @system.setter
    def system(self, new_value):
        if not isinstance(new_value, systems.DynamicalSystem):
            raise TypeError('The dynamical system associated with an integrator'
                            ' must be a \'DynamicalSystem\' instance.')
        self._system = new_value

    def reset(self):
        """Reset the integrator."""
        super().reset()
        self._states = np.zeros((len(self.times), self._state_length()))
        self.states[0] = self.config.initial_state[: self._state_length()]

    def run(self, task, time='forward', verbose=False, catch_escape=False,
            backend='vode', **kwargs):
        """Run the numerical integration.

        Run a numerical integration of the dynamical system using either an
        [ode](https://tinyurl.com/scipy-integrate-ode) instance from SciPy's
        `integrate` module, or a classical Runge--Kutta method.

        Parameters
        ----------
        task : {'ode', 'runge-kutta'}, optional
            Choose to integrate with SciPy's methods or with a classical
            Runge-- Kutta method. Defaults to `'ode'`.
        time : {'forward', 'backward'}, optional
            Chooses if the intergration is performed forward of backward in
            time. If the integration is done backward in time, it is done from
            the initial time given by the configuration and for a total time
            interval of the same length as if it was done forward. Defaults to
            `'forward'`.
        verbose : bool, optional
            If `True`, a progression bar will be printed to show how much of the
            integration has been performed. Defaults to `False`.
        catch_escape : bool, optional
            If `True`,  the integration will stop as soon as a state component
            escapes the interval \\([-1, 1]\\), and the integration will be
            considered to have failed. Defaults to `False`.
        backend : {'vode', 'zvode', 'lsoda', 'dopri5', 'dop853'}, optional
            Integrator used with `ode`. Defaults to `'vode'`. It has no effect
            if `task` is not set to `ode`.
        **kwargs
            Keyword arguments to be passed to the `set_integrator` method of
            the `ode` solver. It has no effect if `task` is not set to `ode`.

        Warns
        -----
        popnet.exceptions.PopNetWarning
            If the integration fails.
        """
        self._success = True
        if time == 'forward':
            def field(t, x): return self._field(t, x)
            def jac(t, x): return self._jac(t, x)
        elif time == 'backward':
            def field(t, x): 
                return - self._field(2 * self.config.initial_time - t, x)
            def jac(t, x): 
                return - self._jac(2 * self.config.initial_time - t, x)
        else:
            raise ValueError(f'Unknown value {time} for \'time\' keyword. Valid'
                             ' values are \'forward\' and \'backward\'.')
        if verbose:
            def progress(rg): return tqdm(rg)
        else:
            def progress(rg): return rg
        if catch_escape:
            def has_escaped(state): return any(np.abs(state) > 1)
        else:
            def has_escaped(state): return False
        if task == 'ode':
            self._run_ode(field, jac, progress, has_escaped, backend, **kwargs)
        elif task == 'runge-kutta':
            self._run_runge_kutta(field, progress, has_escaped)
        else:
            raise ValueError(f'Unknown task {task} for \'Integrator.run\'. Acce'
                             'pted values are \'ode\'  and \'runge-kutta\'.')
        if time == 'backward':
            self._times = np.flip(2 * self.config.initial_time - self.times)
            self._states = np.flip(self.states, axis=0)
        if not self.success:
            warn(f'Integration failed with configuration {self.config.ID}.',
                 category=PopNetWarning, stacklevel=2)

    def _field(self, t, Y):
        """Vector field.

        Vector field corresponding to the studied dynamical system.

        Parameters
        ----------
        t : float
            Current time.
        Y : array_like
            Current state of the network.

        Returns
        -------
        array_like
            Gradient of the vector field evaluated at time `t` and state `Y`.
        """
        return self.system.vector_field(Y)

    def _jac(self, t, Y):
        """Jacobian of the `_field` method.

        Jacobian of the vector field corresponding to the studied dynamical
        system. It does not have to be implemented by subclasses.

        Parameters
        ----------
        t : float
            Current time.
        Y : array_like
            Current state of the network.

        Returns
        -------
        array_like
            Jacobian of the vector field evaluated at time `t` and state `Y`.
        """
        return self.system.jac(Y)

    def _output_states(self):
        """States array to output."""
        if isinstance(self.system, systems.WilsonCowanSystem):
            p = len(self.config.network.populations)
            output_states = np.zeros((len(self.times), 2*p))
            output_states[:,:p] = self.states
            for J, popJ in enumerate(self.config.network.populations):
                output_states[:,p+J] = popJ.beta / popJ.gamma * self.states[:,J]
            return output_states
        return super()._output_states()

    def _run_ode(self, field, jac, progress, has_escaped, backend, **kwargs):
        """Run a numerical integration with `scipy.integrate.ode`."""
        try:
            jac(0, self.states[0])
        except NotImplementedError:
            solver = ode(field)
        else:
            solver = ode(field, jac)
        solver.set_integrator(backend, **kwargs)
        solver.set_initial_value(self.states[0])
        for j in progress(range(1, len(self.times))):
            self.states[j] = solver.integrate(solver.t+self.config.delta)[:]
            if not solver.successful() or has_escaped(self.states[j]):
                self._success = False
                break

    def _run_runge_kutta(self, field, progress, has_escaped):
        """Run a numerical integration with a classical Runge--Kutta method."""
        for j in progress(range(1, len(self.times))):
            k1 = field(0, self.states[j-1])
            k2 = field(0, self.states[j-1] + self.config.delta * k1 / 2)
            k3 = field(0, self.states[j-1] + self.config.delta * k2 / 2)
            k4 = field(0, self.states[j-1] + self.config.delta * k3)
            slope = (k1 + 2*k2 + 2*k3 + k4) / 6
            self.states[j] = self.states[j-1] + self.config.delta * slope
            if has_escaped(self.states[j]):
                self._success = False
                break

    def _state_length(self):
        """Length of the macroscopic states."""
        return self.system.dim


class Simulator(Executor):
    """Numerical simulator of the stochastic process on a network.

    `Simulator` extends `Executor` to perform numerical simulations of a
    stochastic process on a network whose mean field reduction is represented
    macroscopically by the Wilson--Cowan model.

    Parameters
    ----------
    config : popnet.structures.MicroConfiguration
        Sets the configuration used for the simulation.
    act : {'step', 'sigmoid'}, optional
        Sets the shape of the activation rate of a neuron. Defaults to `step`.

    Attributes
    ----------
    config : popnet.structures.MicroConfiguration
        Configuration used for the simulations. See `Simulator.config`.
    states, times : array_like
        Macroscopic state of the network with respect to time. See
        `Simulator.states` and `Simulator.times`.
    micro_states, transition_times : array_like
        Microscopic state of the network with respect to time. See
        `Simulator.micro_states` and `Simulator.transition_times`.
    activation_rates : list
        Activation rates of the neurons of the network. See
        `Simulator.activation_rates`.
    activation_rates_shape : {'step', 'sigmoid'}
        Describes the shape of the activation rate of neurons of the network.
        See `Simulator.activation_rates_shape`.

    """

    def __init__(self, config, act='step', **kwargs):
        self.activation_rates_shape = act
        super().__init__(config, **kwargs)
        try:
            output_type = OUTPUT_TYPES[type(self)]
        except KeyError:
            warn('The type of simulator you want to run is not recognized by '
                 'PopNet. The simulation might be possible if a \'run\' method '
                 'is implemented, but it will not be possible to output the '
                 'results with \'Simulator.output\'.', category=PopNetWarning,
                 stacklevel=2)
        else:
            self._output_type = output_type

    @Executor.config.setter
    def config(self, new_value):
        if not isinstance(new_value, structures.MicroConfiguration):
            raise TypeError('The configuration used with a simulator must be a '
                            '\'MicroConfiguration\' instance.')
        self._config = new_value
        self.reset()

    @property
    def micro_states(self):
        """Microscopic state of the network with respect to time.

        Microscopic state of the network at each time step given by
        `Simulator.transition_times`. It does not contain any relevant data at
        initialization or right after a reset, but it is updated during a call
        to `Simulator.run`. It cannot be manually set nor deleted.
        """
        return self._micro_states

    @property
    def transition_times(self):
        """Time.

        Times at which transitions have occurred for a given trajectory. Unlike
        `Simulator.times`, it is not set according to the configuration used,
        but rather updated stochastically during a call to `Simulator.run`. It
        does not contain any relevant data at initialization or right after a
        reset. It cannot be manually set nor deleted.
        """
        return self._transition_times

    @property
    def activation_rates(self):
        """Activation rates of the neurons.

        List of functions representing the activation rates of the network's
        neurons. `activation_rates[j](x)` gives the activation rate of the
        *j*th neuron of the network if the state of the whole network is `x`.
        It cannot be manually set nor deleted.
        """
        return self._activation_rates

    @property
    def activation_rates_shape(self):
        """Shape of the activation rates.

        Shape of the activation rate of a single neuron of the network as a
        function of its input. The only valid values are:

            - `'step'`. In that case, the activation rate is a step function
              going from zero to alpha at the neuron's threshold `theta`.
            - `'sigmoid'`. In that case, the activation rate is the logistic
              function `popnet.structures.Population.F` of the population to
              which belongs the neuron.

        !!! note
            After initialization, a change in the value of this property will
            only have an effect after a reset of the simulator with
            `Simulator.reset`.

        Warnings
        --------
        It is important to understand that the values taken by this parameter
        yield two related, but distinct, interpretations of the models studied
        in this package. Indeed, the macroscopic models can be seen as
        approximations of two microscopic models: one in which neurons can
        activate at a constant rate when given sufficient input but cannot
        otherwise, and another one in which neurons rather activate at a rate
        depending on the input according to a sigmoid function. While both of
        these interpretations are valid, the documentation of this package is
        written assuming the former interpretation. Thus, if simulations are
        performed using the `'sigmoid'` option, the interpretation of the
        function `popnet.structures.Population.F` changes a little bit. This
        should be adapted shortly.
        """
        return self._activation_rates_shape

    @activation_rates_shape.setter
    def activation_rates_shape(self, new_value):
        valid_values = ['step', 'sigmoid']
        if new_value not in valid_values:
            raise ValueError(f'Unexpected value {new_value} for the shape of '
                             'the activation rates. Valid values are '
                             f'{valid_values}.')
        self._activation_rates_shape = new_value

    def calcium_output(self, indices=None, growth_rate=None, decay_rate=None):
        """Get the calcium concentration in neural cells.

        Get the concentration of calcium in neural cells with respect to time. 

        Parameters
        ----------
        indices : int or array_like, optional
            Indices of neurons for which to get the calcium concentration.
            Defaults to `None`, in which case the calcium concentration is given
            for every neuron of the network.
        growth_rate : float, optional
            Initial growth rate of the calcium concentration. It must be
            positive. Defaults to `None`, in which case it is replaced with the
            inverse of the configuration's time step
            `popnet.structures.Configuration.delta`.
        decay_rate : float, optional
            Decay rate of the calcium concentration. It must be positive, and it
            should be much smaller than the initial growth rate. Defaults to
            `None`, in which case it is replaced with five percent of the
            initial growth rate.

        Returns
        -------
        array_like
            Calcium concentration with respect to time for every requested
            neuron, with neurons along the first axis and time along the second.
            If a single neuron was requested, it will be one-dimensional.

        Raises
        ------
        ValueError
            If `indices` is not a valid list of indices for neurons of the
            network.
        """
        if growth_rate is None:
            growth_rate = 1 / self.config.delta
        if decay_rate is None:
            decay_rate = .05 * growth_rate
        if isinstance(indices, int):
            return self._get_calcium_output(indices, growth_rate, decay_rate)
        valid_indices = np.arange(self.config.network.size())
        if indices is None:
            indices = valid_indices
        try:
            valid_indices[indices]
        except IndexError as error:
            raise ValueError(f'{indices} is not a valid list of indices for '
                             'neurons of the network.') from error
        calcium = np.zeros((N := len(indices), len(self.transition_times)))
        for j in range(N):
            calcium[j,:] = self._get_calcium_output(j, growth_rate, decay_rate)
        return calcium

    def close(self):
        """Delete all data attributes of the simulator."""
        super().close()
        del self._micro_states
        del self._transition_times

    def micro_output(self, fmt='ternary'):
        """Get the simulation's microscopic output.
        
        Get the microscopic state of the network with respect to time from the
        last simulation that was performed.

        Parameters
        ----------
        fmt : {'binary', 'ternary', 'calcium'}, optional
            Format of the neurons' states. If `'ternary'`, a neuron's state can
            take the values `1`, `1j` or `0`, associated with the *active*,
            *refractory* and *sensitive* states respectively. If `'binary'`, a
            neuron's state can take the values `1` or `0`, where `1` is still
            associated with the active state, but `0` is rather associated with
            any non-active state (sensitive or refractory). If `'calcium'`,
            the returned output is the default given by
            `Simulator.calcium_output`. Defaults to `'ternary'`.

        Returns
        -------
        array_like
            Microscopic state of the network with respect to time.

        Raises
        ------
        ValueError
            If `fmt` is passed an unexpected value.
        """
        self._check_if_run()
        if fmt == 'ternary':
            return self.micro_states
        if fmt == 'binary':
            return np.real(self.micro_states)
        if fmt == 'calcium':
            return self.calcium_output()
        raise ValueError(f'Unknown format {fmt} for microscopic states.')

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self._states = None
        self._transition_times = [self.config.initial_time]
        self._micro_states = [self.config.micro_initial_state.copy()]
        self._reset_activation_rates()

    def single_run(self, rng, do_step, iterate):
        """Run a single simulation.

        Run a simulation to obtain a possible trajectory of the stochastic
        process which describes the evolution of the network. To obtain this
        trajectory, the Doob--Gillespie algorithm is used either with the
        direct method or with the first reaction method. See the
        [Notes](#simulator-single-run-notes) section below for more details
        about the algorithm.

        !!! note
            The recommended way to perform simulations of the stochastic process
            is *not* to use this method, but rather to use `SimpleSimulator.run`
            or `ChainSimulator.run`, which both use it internally.

        Parameters
        ----------
        rng : numpy.random.Generator
            A random number generator.
        do_step : callable
            Dictates how to do the Monte Carlo step of the Doob--Gillespie
            algorithm. To be passed to `iterate`. It expects as inputs, in
            order: `rng`, the current time `t`, an array of the next possible
            network states, and an array of the corresponding transition rates.
            It should return the index of the next network state and the time
            interval between `t` and the next transition.
        iterate : callable
            Dictates how a complete iteration of the simulation is performed.
            This includes the Monte Carlo step as well as all other tasks that
            should be done at each time step. It expects as inputs, in order:
            `do_step`, `rng`, `t` and `x`, where `t` and `x` are the current
            time and network state. It should return the next time and network
            state.

        Notes {#simulator-single-run-notes}
        -----
        From the microscopic point of view, the evolution of the state of the
        whole network is described by a stochastic process. The simulation run
        by this method outputs a possible trajectory of this stochastic process,
        using the Doob--Gillespie algorithm, popularized by Gillespie in [3] and
        based on results of Doob [1,2]. The idea to pass from a state to another
        is first to find all of the states to which the network can go from the
        current one, with the corresponding transition rates. This information
        is in fact sufficient to determine the distribution of the time at which
        the next transition occur, and which one will occur. 

        In [3], Gillespie introduces two methods, called the *direct* and
        *first reaction* methods respectively, to choose the time interval until
        the next transition and the next state of the system.

         - **Direct method.** First, the total transition rate out of the
           current state is computed, and a time interval until the next
           transition is taken randomly knowing that it is exponentially
           distributed with parameter equal to this total out rate. Then a next
           state is chosen randomly knowing that the probability of going to a
           given other state is proportional to the corresponding transition
           rate.

         - **First reaction method.** For every possible next state, a time at
           which the corresponding transition could occur is randomly generated,
           knowing that this time is exponentially distributed with parameter
           equal to the transition rate. The transition that should occur first
           is chosen, and the state is updated accordingly.

        References
        ----------
         1. Doob, J. L. “Topics in the Theory of Markoff Chains.” *Transactions
            of the American Mathematical Society* **52**, 37--64 (1942).
            doi: [10.2307/1990152](https://doi.org/10.2307/1990152).
         2. Doob, J. L. “Markoff Chains--Denumerable Case.” *Transactions of the
            American Mathematical Society* **58**, 455--473 (1945).
            doi: [10.2307/1990339](https://doi.org/10.2307/1990339).
         3. Gillespie, D. T. “A General Method for Numerically Simulating the
            Stochastic Time Evolution of Coupled Chemical Reactions.” *Journal
            of Computational Physics* **22**, 403--434 (1976). doi:
            [10.1016/0021-9991(76)90041-3](
            https://doi.org/10.1016/0021-9991(76)90041-3).
        """
        t = self.transition_times[0]
        x = self.micro_states[0]
        while t < self.config.final_time:
            t, x = iterate(do_step, rng, t, x)
        self._micro_states = np.array(self.micro_states)
        self._transition_times = np.array(self.transition_times)
        self._update_states()

    def _check_sizes(self):
        """Check the consistency of the network's size and the initial state."""
        if len(self.config.micro_initial_state) != self.config.network.size():
            raise PopNetError('It seems that the size of the network has '
                              'changed since the microscopic initial state has '
                              'been set. It has to be reset. The network\'s '
                              'parameters might also have to be reset.')

    def _direct_method(self, rng, t, next_states, rates):
        """Obtain the next state and time from the direct method."""
        out_rate = np.sum(rates)
        threshold_rate = rng.random() * out_rate
        j = 0
        sum_of_rates = rates[0]
        while sum_of_rates < threshold_rate:
            sum_of_rates += rates[j+1]
            j += 1
        return j, (1 / out_rate) * np.log(1 / rng.random())

    def _first_reaction_method(self, rng, t, next_states, rates):
        """Obtain the next state and time from the first reaction method."""
        next_times = (1 / rates) * np.log(1 / rng.random(len(rates)))
        j = np.argmin(next_times)
        return j, next_times[j]

    def _get_calcium_output(self, j, growth_rate, decay_rate):
        """Get the calcium concentration in neuron *j* with respect to time."""
        binary_state = np.real(self.micro_states[:,j])
        all_activation_indices = np.nonzero(binary_state)[0]
        activation_indices = [k for i,k in enumerate(all_activation_indices)
                              if all_activation_indices[i-1] != k-1]
        calcium = np.zeros((len(activation_indices), 
                            len(self.transition_times)))
        for k, activation_index in enumerate(activation_indices):
            t = self.transition_times[activation_index:] 
            t0 = self.transition_times[activation_index]
            calcium[k,activation_index:] = ((1 - np.exp(-growth_rate * (t-t0)))
                                                * np.exp(-decay_rate * (t-t0)))
        calcium = np.sum(calcium, axis=0)
        return calcium

    def _iterate(self, do_step, rng, t, x):
        """Perform a single iteration of a simulation.

        Perform a single iteration of a simulation. 

        Parameters
        ----------
        do_step : function
            Dictates how the Monte Carlo step of Gillespie's algorithm.
        rng : Generator
            Random number generator.
        t : float
            Current time step.
        x : array_like
            Current state of the network.

        Returns
        -------
        float
            Next time step.
        array_like
            Next state of the network.
        """
        next_states, rates = self._next_states_and_rates(x)
        j, time_interval = do_step(rng, t, next_states, rates)
        x = next_states[j].copy()
        t += time_interval
        self.transition_times.append(t)
        self.micro_states.append(x.copy())
        return t, x

    def _make_sigmoid_activation_rate(self, j, J):
        """Define a sigmoid activation rate for the `j`th neuron."""
        def act(x):
            b = np.dot(self.config.network.W[j], np.real(x)) + self.config.Q[J]
            F_value = self.config.network.populations[J].F(b)
            return self.config.network.alpha[j] * F_value
        return act

    def _make_step_activation_rate(self, j, J):
        """Define a step activation rate for the `j`th neuron."""
        def act(x):
            b = np.dot(self.config.network.W[j], np.real(x)) + self.config.Q[J]
            if b < self.config.network.theta[j]:
                return 0.
            else:
                return self.config.network.alpha[j]
        return act

    def _make_activation_rate(self, j):
        """Define the activation rate function for the `j`th neuron."""
        J = 0
        sum_sizes = self.config.network.populations[0].size
        while j > sum_sizes:
            sum_sizes += self.config.network.populations[J+1].size
            J += 1
        if self.activation_rates_shape == 'step':
            return self._make_step_activation_rate(j, J)
        elif self.activation_rates_shape == 'sigmoid':
            return self._make_sigmoid_activation_rate(j, J)
        raise ValueError(f'Unexpected value {self.activation_rates_shape} for '
                         'the shape of the activation rate function.')

    def _next_states_and_rates(self, x):
        """Get all possible states to which the network can go from `x`.

        Knowing that the network is in state `x`, get all states which are 
        accessible next with the rates associated with each possible transition.

        Returns
        -------
        tuple of array_like
            The next possible states, and the associated transition rates. Both
            arrays are arranged so that the *j*th element of the transition rate
            vector is the rate at which the network can make a transition to the
            state corresponding to the *j*th row of the array of next states.
        """
        next_states = np.resize(x, (N := self.config.network.size(), N))
        rates = np.zeros(N)
        for j in range(N):
            next_states[j,j], rates[j] = self._next_state_and_rate(j, x)
        return next_states, rates

    def _next_state_and_rate(self, j, x):
        """Get the next state and transition rate of the `j`th neuron.

        Knowing that the network is in state `x`, get the next accessible state
        of `j`th neuron, with the rate at which this neuron will make a 
        transition. 

        Parameters
        ----------
        j : int
            The neuron for which to get the next state and transition rate.
        x : array_like
            The current state of the network.

        Returns
        -------
        tuple of complex and float
            The next state of the `j`th neuron with associated transition rate. 

        Raises
        ------
        ValueError
            If the `j`th neuron is in a non valid state. 
        """
        if (z := x[j]) == 0.:
            return 1., self.activation_rates[j](x)
        if z == 1.:
            return 1j, self.config.network.beta[j]
        if z == 1j:
            return 0., self.config.network.gamma[j]
        raise ValueError('The state of a neuron should always be 0, 1 or the'
                         'imaginary unit.')

    def _reset_activation_rates(self):
        """Reset the activation rate functions."""
        self._activation_rates = [self._make_activation_rate(j)
                                  for j in range(self.config.network.size())]

    def _state_length(self):
        """Length of the macroscopic states."""
        return 2 * len(self.config.network.populations)

    def _update_states(self):
        """Update `states` based on `micro_states`.

        Compute the macroscopic states of the network from `micro_states`, and
        update `states` in consequence.
        """
        p = len(self.config.network.populations)
        states = np.zeros((len(self.transition_times), 2*p))
        j = 0
        for J, popJ in enumerate(self.config.network.populations):
            states[:,J]   = np.sum(np.real(self.micro_states[:,j:j+popJ.size]), 
                                axis=1) / popJ.size
            states[:,p+J] = np.sum(np.imag(self.micro_states[:,j:j+popJ.size]), 
                                axis=1) / popJ.size
            j += popJ.size
        self._states = states


class SimpleSimulator(Simulator):
    """Perform single simulations of the stochastic process on a network.

    `SimpleSimulator` extends `Simulator` to ease the task of running single
    simulations of the stochastic process. It has dedicated methods to run
    simulations and output a `popnet.graphics.Trajectory` instance. Its data
    attributes are the same as in the base class.

    The initialization parameters are the same as in the base class, except that
    the output type does not have to be explicitely given.

    """

    def run(self, method='direct', verbose=False):
        """Run a simulation.

        Run a simulation to obtain a possible trajectory of the stochastic
        process which describes the evolution of the network. To obtain this
        trajectory, we use the Doob--Gillespie algorithm, either with the direct
        or with the first reaction method. See `Simulator.single_run` for more
        details about the Doob--Gillespie algorithm.

        Parameters
        ----------
        method : {'direct', 'first reaction'}, optional
            Chooses which method is used to perform the Monte Carlo step in the
            Doob--Gillespie algorithm. Defaults to `'direct'`.
        verbose : bool, optional
            If `True`, the current time will be printed. Defaults to `False`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the length of the microscopic initial state is different of the
            network's size.
        ValueError
            If an unexpected value is passed to `method`.
        """
        self._check_sizes()
        if method == 'direct':
            def do_step(rng,t,ns,r): return self._direct_method(rng,t,ns,r)
        elif method == 'first reaction':
            def do_step(rng,t,ns,r): return self._first_reaction_method(rng,t,ns,r)
        else:
            raise ValueError(f'Simulator does not know a method {method}.')
        if verbose:
            def iterate(do_step, rng, t, x):
                print(f't = {t:<.2f}', end='\r')
                return self._iterate(do_step, rng, t, x)
        else:
            def iterate(do_step, rng, t, x): 
                return self._iterate(do_step, rng, t, x)
        self.single_run(np.random.default_rng(), do_step, iterate)
        if verbose:
            print(10*' ', end='\r')
            print('Done!')
        self._success = True

    def save_output(self, name=None, folder=None):
        """Save the simulation's output to a text file.

        Extends the base class method by saving additionally the times at which
        transitions occur. This is done by saving the array
        `Simulator.transition_times` under *ID - name (times).txt*, where *ID*
        is the ID of the configuration used for the simulation, and *name* is
        `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved output. Defaults to `None`, in which case
            it is replaced with `'Trajectory'`.
        folder : str, optional
            A folder in which the files are saved. If it does not exist in the
            current directory, it is created. Defaults to `None`, in which case
            the files are saved in the current directory.

        Returns
        -------
        name : str
            Name given to the saved output.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the simulation has not been performed yet.
        """
        name = super().save_output(name=name, folder=folder)
        filename = _internals._format_filename(folder, self.config.ID,
                                               f'{name} (times)')
        np.savetxt(filename, self.transition_times, fmt='%+.12f')
        return name

    def _output_times(self):
        """Times array to output."""
        return self.transition_times


class ChainSimulator(Simulator):
    """Simulate multiple times the stochastic process on a network.

    `ChainSimulator` extends `Simulator` to ease the task of running many
    simulations of the stochastic process on the same network with the same
    configuration in order to obtain statistics. It has dedicated methods to
    run many simulations and output a `popnet.graphics.Statistics` instance.
    Its data attributes are the same as in the base, except for a new
    `ChainSimulator.samples`, which stores the trajectories obtained from
    simulations of the stochastic process.

    The initialization parameters are the same as in the base class, except that
    the output type does not have to be explicitely given.

    """

    @property
    def samples(self):
        """Samples of trajectories.

        Samples of trajectories of the stochastic process, once a simulation has
        been performed. It is a three dimensional array, where the first axis is
        time, the second is the macroscopic state component, and the third is
        associated with a given trajectory. It cannot be manually set nor deleted.
        """
        return self._samples

    def close(self):
        """Delete all data attributes of the simulator."""
        super().close()
        del self._samples

    def reset(self):
        """Reset the simulator."""
        super().reset()
        self._samples = None

    def run(self, method='direct', initial_state='fixed', verbose=False):
        """Run multiple simulations.

        Run multiple simulations of the stochastic process which describes the
        evolution of the network, in order to obtain a sample of possible
        trajectories. Each single simulation is perforfmed with the
        Doob--Gillespie algorithm, either with the direct or the first reaction
        method. See `Simulator.single_run` for more details about the
        Doob--Gillespie algorithm.

        Parameters
        ----------
        method : {'direct', 'first reaction'}, optional
            Chooses which method is used to perform the Monte Carlo step in the
            Doob--Gillespie algorithm. Defaults to `'direct'`.
        initial_state : {'fixed', 'random'}, optional
            If `'fixed'`, the microscopic initial state is the same in all
            simulations. If `'random'`, the microscopic initial state is reset
            randomly with
            `popnet.structures.MicroConfiguration.reset_micro_initial_state`
            between each simulation. Defaults to `'fixed'`.
        verbose : bool, optional
            If `True`, a progression bar will be printed to show how much of the
            `n` simulations have been performed. Defaults to `False`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the length of the microscopic initial state is different of the
            network's size.
        ValueError
            If an unexpected value is passed to `method`.
        """
        self._check_sizes()
        samples = np.zeros((1+self.config.iterations, self._state_length(), 
                            self.config.executions))
        if method == 'direct':
            def do_step(rng,t,ns,r): return self._direct_method(rng,t,ns,r)
        elif method == 'first reaction':
            def do_step(rng,t,ns,r): return self._first_reaction_method(rng,t,ns,r)
        else:
            raise ValueError(f'\'Simulator\' does not know a method {method}.')
        if initial_state == 'fixed':
            def reset_initial_state(): pass
        elif initial_state == 'random':
            def reset_initial_state(): self.config.reset_micro_initial_state()
        else:
            raise ValueError(f'Unknown keyword {initial_state} to choose the '
                             'initial state. Valid values are \'fixed\' and '
                             '\'random\'.')
        if verbose:
            def progress(rg): return tqdm(rg)
        else:
            def progress(rg): return rg
        rng = np.random.default_rng()
        for j in progress(range(self.config.executions)):
            self.single_run(rng, do_step, self._iterate)
            for J in range(self._state_length()):
                samples[:,J,j] = np.interp(self.times, self.transition_times, 
                                           self.states.T[J])
            reset_initial_state()
            self.reset()
        self._samples = samples
        self._success = True

    def save_output(self, name=None, folder=None):
        """Save the samples obtained from simulations.

        Overrides the base class method to save the samples of trajectories
        obtainedfrom numerical simulations rather than the macroscopic states
        they yield. Each state component *X* is saved in its own file, named
        *ID - name X.txt*, where *ID* is the ID of the configuration used for
        the simulations, and *name* is `name`.

        Parameters
        ----------
        name : str, optional
            Name to give to the saved samples. Defaults to `None`, in which case
            it is replaced with `'Sample'`.
        folder : str, optional
            A folder in which the files are saved. If it does not exist in the
            current directory, it is created. Defaults to `None`, in which case
            the files are saved in the current directory.

        Returns
        -------
        name : str
            Name given to the saved samples.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the simulations have not been performed yet.
        """
        self._check_if_run()
        name = self._output_type._get_sample_name(name)
        _internals._make_sure_folder_exists(folder)
        for J, X in enumerate(self.config._variables[:self._state_length()]):
            filename = _internals._format_filename(folder, self.config.ID,
                                                   f'{name} {X}')
            h = (f'In each column are the values of {X} with respect to time '
                 'for a given trajectory.')
            np.savetxt(filename, self.samples[:,J,:], fmt='%+.12f', header=h)
        return name

    def _output_states(self):
        """States array to output."""
        return self.samples


OUTPUT_TYPES = {systems.WilsonCowanSystem: graphics.Solution,
                systems.MeanFieldSystem: graphics.Solution,
                systems.MixedSystem: graphics.Solution,
                systems.TaylorExtendedSystem: graphics.ExtendedSolution,
                systems._TaylorExtendedSystemOne: graphics.ExtendedSolution,
                systems.ExtendedSystem: graphics.ExtendedSolution,
                SimpleSimulator: graphics.Trajectory,
                ChainSimulator: graphics.Statistics}
"""Type in which an execution should be output, indexed by
`popnet.systems.DynamicalSystem` or by `Simulator` subclass. This dictionary
is intended to be used only when more entries should be added, so that new
dynamical systems and new simulators can be defined by a user."""


@singledispatch
def get_integrator(arg, system_name=None, **kwargs):
    """Get a numerical integrator.

    Define a numerical integrator, either from a dynamical system or from a
    configuration and a keyword specifying which dynamical system should be
    integrated.

    Parameters
    ----------
    arg : popnet.systems.DynamicalSystem or popnet.structures.Configuration
        Either a dynamical system to integrate, or a configuration to associate
        to the integrator.
    system_name : str, optional
        Decides which dynamical system is integrated when `arg` is a
        configuration. It is mandatory when `arg` is a configuration. but has no
        effect when it is a dynamical system. The following values are accepted.

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
    Integrator
        Integrator initialized with given parameters.

    Raises
    ------
    popnet.exceptions.PopNetError
        If `system_name` is given a non-valid value.
    TypeError
        If `arg` is neither a `popnet.structures.Configuration` instance nor a
        `popnet.systems.DynamicalSystem` instance.
    """
    raise TypeError('\'get_integrator\' expects its first argument to be either'
                    ' a \'Configuration\' or a \'DynamicalSystem\' instance.') 


@get_integrator.register(structures.Configuration)
def _(config, system_name=None, **kwargs):
    system = systems.get_system(config, system_name=system_name)
    return Integrator(system, **kwargs)


@get_integrator.register(systems.DynamicalSystem)
def _(system, system_name=None, **kwargs):
    return Integrator(system, **kwargs)



def get_simulator(config, act='step', mode='individual'):
    """Get a simulator to perform stochastic simulations.

    Define a simulator in order to perform either individual simulations, or
    chains of simulations.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration to associate with the simulator.
    act : {'step', 'sigmoid'}, optional
        Shape of neurons' activation rates. If `'step'`, an activation rate is
        a step function going from zero to `alpha` at the threshold `theta`. If
        `'sigmoid'`, an activation rate is the logistic function
        `popnet.structures.F` of the population to which belongs a neuron.
        Defaults to `'step'`.
    mode : {'individual', 'chain'}, optional
        How the simulations should be executed. If `'individual'`, the simulator
        will be defined to run one simulation at a time. If `'chain'`, it will
        be defined to run a sequence of simulations at every run.

    Returns
    -------
    SimpleSimulator or ChainSimulator
        Simulator initialized with given configuration.

    Raises
    ------
    popnet.exceptions.PopNetError
        If `mode` is given an unexpected value.
    """
    if mode == 'individual':
        return SimpleSimulator(config, act=act)
    elif mode == 'chain':
        return ChainSimulator(config, act=act)
    raise PopNetError(f'Unknown execution mode {mode}.')

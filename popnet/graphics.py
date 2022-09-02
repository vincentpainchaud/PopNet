"""Functions to draw figures representing results of numerical experiments.

This module defines several classes to provide a consistent interface to
produce [Matplotlib](https://matplotlib.org/) figures to represent results
obtained from numerical experiments performed with the rest of the package. The
classes it defines are briefly described in the
[Classes And Hierarchy](#classes-and-hierarchy) section below.

The module also defines a function `figure`, which allows to initialize a
Matplotlib figure with a format consistent with that used by the module's
classes, and a function `draw` to draw such figures. Finally, it introduces
functions to load results from numerical experiments performed with the
`popnet.executors` module, when these have been saved using
`popnet.executors.Executor.save_output`.

Classes and hierarchy
---------------------
The important classes of the module are summarized below. The indentation
follows the hierarchy. 

 - `Graphics` : General base class to plot diagrams.
     - `PhasePlane` : Draw phase planes of dynamical systems.
     - `Result` : Represent results of numerical experiments.
         - `Solution` : Outputs of numerical integrations where covariances are
           not included.
             - `ExtendedSolution` : Outputs of numerical integrations where
               covariances are included.
         - `Trajectory` : Outputs of individual simulations.
         - `Statistics` : Outputs of series of simulations.
         - `Spectrum` : Fourier transform of another result.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import root
from warnings import warn

from .exceptions import *
from . import _internals
from . import structures
from . import systems


class Graphics:
    """General base class to plot various diagrams.

    This class provides basic tools to plot various diagrams related to
    numerical experiments performed by PopNet. It relies heavily on the
    [Matplotlib](https://matplotlib.org/) library. This class is intended to be
    used mainly through its subclasses.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration associated with the plot.
    name : str, optional
        Name to associate with the plot. Defaults to `None`, in which case it
        is replaced with `Graphics.default_name`.

    Attributes
    ----------
    config : popnet.structures.Configuration
        Configuration associated with the plot. See `Graphics.config`.
    name : str
        Name associated with the graphics. See `Graphics.name`.
    fig : matplotlib.figure.Figure
        A Matplotlib figure. See `Graphics.fig`.
    ax : matplotlib.axes.Axes
        Axes of `fig`. See `Graphics.ax`.

    """

    default_name = 'Graphics'
    """Default name given to instances."""

    def __init__(self, config, name=None):
        if not isinstance(config, structures.Configuration):
            raise TypeError('The configuration used with a \'Graphics\' '
                            'instance must be a \'Configuration\' instance.')
        self._config = config
        self.name = self._get_name(name)
        self.fig = None
        self.ax = None

    @property
    def config(self):
        """Configuration associated with the figure.

        Configuration associated with the figure. It is set at initialization,
        and cannot be set or deleted afterwards.
        """
        return self._config

    @property
    def name(self):
        """Name of the figure. It has to be a string."""
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError(f'Graphics.name must be a string.')
        self._name = new_name

    @property
    def fig(self):
        """A figure on which to draw various plots.

        A [`matplotlib.figure.Figure`](https://31c8.short.gy/mpl-figure-Figure)
        object that can be used to draw plots. This is the figure where
        `Graphics`' methods can plot curves. It is set automatically when
        `Graphics.activate` is called. It cannot be deleted manually.
        """
        return self._fig

    @fig.setter
    def fig(self, new_value):
        if new_value is None:
            pass
        elif not isinstance(new_value, mpl.figure.Figure):
            raise TypeError('Graphics.fig must be a '
                            '\'matplotlib.figure.Figure\' instance.')
        self._fig = new_value

    @property
    def ax(self):
        """Axes of the current figure.

        A [`matplotlib.axes.Axes`](https://31c8.short.gy/mpl-axes-Axes) object
        correponding to the axes of `Graphics.fig`. It is set automatically
        when `Graphics.activate` is called. It cannot be deleted manually.
        """
        return self._ax

    @ax.setter
    def ax(self, new_value):
        if new_value is None:
            pass
        elif not isinstance(new_value, mpl.axes.Axes):
            raise TypeError(f'Graphics.ax must be a '
                            '\'matplotlib.axes.Axes\' instance.')
        self._ax = new_value

    def activate(self, figsize=(5,3.75), dpi=150, tight_layout=True, 
                 font_family='serif', usetex=False, preamble=None, **kwargs):
        """Activate a figure.
        
        Create a Matplotlib figure to plot diagrams, and set `Graphics.fig` and
        `Graphics.ax` to refer to the
        [`matplotlib.figure.Figure`](https://31c8.short.gy/mpl-figure-Figure)
        and [`matplotlib.axes.Axes`](https://31c8.short.gy/mpl-axes-Axes)
        objects corresponding to this figure. The figure is initialized with
        the default formatting defined by `figure`: in fact, if `graphic` is a
        `Graphics` object, the calls
        
        >>> graphic.activate(**kwargs)
        
        and
        
        >>> graphic.fig, graphic.ax = figure(subplots=None, **kwargs)
        
        are equivalent.

        Parameters
        ----------
        figsize : tuple of float, optional
            Width and height of the figure in inches. Defaults to (5, 3.75).
        dpi : int
            Resolution of the figure in dots per inches. Defaults to 150.
        tight_layout : bool, optional
            Adjust automatically the padding between and aroung subplots using
            [`matplotlib.figure.Figure.tight_layout`](
            https://31c8.short.gy/mpl-tight-layout). Defaults to `True`.
        font_family : {'serif', 'sans-serif'}, optional
            Determines if a serif or sans serif font family is used. Defaults
            to `'serif'`.
        usetex : bool, optional
            Determines if LaTeX is used to draw the figure. Defaults to `False`.
        preamble : str, optional
            LaTeX preamble when `usetex` is `True`, in which case it case be
            used to load font packages. It has no effect when `usetex` is
            `False`. Defaults to `None`, in which case a default preamble is
            added.
        **kwargs
            Keyword arguments to be passed to
            [`matplotlib.pyplot.figure`](https://31c8.short.gy/plt-figure).
        """
        if 'subplots' in kwargs:
            raise TypeError('Graphics.activate() got an unexpected keyword '
                            'argument \'subplots\'')
        self.fig, self.ax = figure(
            figsize=figsize, dpi=dpi, tight_layout=tight_layout,
            font_family=font_family, usetex=usetex, preamble=preamble, **kwargs)

    def draw(self, name=None, show=True, savefig=False, folder=None, 
             format=None, **kwargs):
        """Draw the plot.

        Draw a figure activated with `Graphics.activate`. If the figure is
        saved, it is named *ID - name*, where *ID* is the configuration's ID,
        *name* is `name`, and has the file format chosen with `format`.

        Parameters
        ----------
        name : str, optional
            Name to give to the figure if saved. Defaults to `None`, in which
            case `Graphics.name` is used.
        show : bool, optional
            Decides if the figure is shown or not. Defaults to `True`.
        savefig : bool, optional
            Decides if the figure is saved or not. Defaults to `False`.
        folder : str, optional
            A folder in which the figure can be saved. If it does not exist in
            the current directory and the figure is saved, it is created.
            Defaults to `None`, in which case the figure is saved in the
            current directory.
        format : str, optional
            The file format under which the figure is saved if `savefig` is
            `True`. It must be a format handled by Matplotlib, which includes
            'png', 'jpg', 'pdf' and 'svg'. Defaults to `None`, in which case
            the file format is Matplotlib's `savefig.format` parameter, which
            defaults to 'png'.
        **kwargs
            Keyword arguments passed to
            [`matplotlib.pyplot.savefig`](https://31c8.short.gy/plt-savefig)
            when `savefig` is `True`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If the figure has not been activated yet.
        """
        self._check_if_activated()
        if savefig:
            if name is None:
                name = self.name
            filename = _internals._format_filename(
                            folder, self.config.ID, name, extension=format)
            _internals._make_sure_folder_exists(folder)
            plt.savefig(filename, format=format, **kwargs)
        if show:
            plt.show()
        plt.close(self.fig)

    def legend(self, lw=2, fontsize=10, ncol=None, handletextpad=0.5, **kwargs):
        """Generate a legend for the figure.

        Generate a legend for the figure with default options.

        Parameters
        ----------
        lw : float, optional
            Line width of the legend handles. Defaults to 2.
        fontsize : float, optional
            Fontsize of the legend labels. Defaults to 10.
        ncol : int, optional
            Number of columns of the legend. Defaults to `None`, in which case
            it is set to the number of populations of the configuration's
            network.
        handletextpad : float, optional
            Padding of the labels. Defaults to 0.5.
        **kwargs
            Keyword arguments to be passed to the
            [`legend`](https://31c8.short.gy/ax-legend) method of
            `Graphics.ax`, which actually adds the legend on the figure.

        Returns
        -------
        matplotlib.legend.Legend
            The figure's legend.
        """
        if ncol is None:
            ncol = len(self.config.network.populations)
        leg = self.ax.legend(fontsize=fontsize, ncol=ncol, 
                             handletextpad=handletextpad, **kwargs)
        for lego in leg.legendHandles:
            lego.set_linewidth(lw)
        return leg

    def _check_if_activated(self):
        """Check if the figure is already activated."""
        if self.fig is None or self.ax is None:
            raise PopNetError('The figure must be activated before to be '
                              'drawn. Call Graphics.activate() first.')

    @classmethod
    def _get_name(cls, name):
        """Return `name`, or the default name if `name` is `None`."""
        if name is None:
            return cls.default_name
        return name


class PhasePlane(Graphics):
    """Draw phase planes for dynamical systems.

    This class is dedicated to plot phase planes of dynamical systems
    implemented in `popnet.systems`. It should be emphasized that this class is
    intended to draw phase *planes*, not one- or three-dimensional phase spaces.

    Parameters
    ----------
    system : popnet.systems.DynamicalSystem
        Dynamical system for which to draw a phase plane.
    axes : tuple of int
        Components of the dynamical system that will be independant variables
        on the phase plane.
    fixed_axes : array_like
        Values of the remaining components other than the independent variables
        corresponding to `axes`.
    name : str, optional
        A name associated with the phase plane. Defaults to `None`, in which
        case it is replaced with `PhasePlane.default_name`.

    Attributes
    ----------
    config : popnet.structures.Configuration
        The configuration associated with the phase plane. See
        `PhasePlane.config`.
    name : str
        Name associated with the phase plane. See `PhasePlane.name`.
    fig : matplotlib.figure.Figure
        A Matplotlib figure. See `PhasePlane.fig`.
    ax : matplotlib.axes.Axes
        The axes of `fig`. See `PhasePlane.ax`.
    system : popnet.systems.DynamicalSystem
        Dynamical system for which to draw a phase plane. See
        `PhasePlane.system`.
    axes : tuple of int
        Components chosen as independant variables for the phase plane. See
        `PhasePlane.axes`.
    fixed_axes : array_like
        Values of the remaining state components other than independent ones.
        See `PhasePlane.fixed_axes`.

    Raises
    ------
    TypeError
        If `system` is not a `popnet.systems.DynamicalSystem` instance.

    Notes
    -----
    The cases where the dynamical system studied is the
    `popnet.systems.MeanFieldSystem` or the `popnet.systems.WilsonCowanSystem`
    are internally handled by private subclasses `_PhasePlaneMeanField` and
    `_PhasePlaneWilsonCowan`, which are automatically instantiated by the class
    constructor of `PhasePlane` when appropriate. The subclasses mentioned
    above are responsible to define the methods used to compute nullclines for
    the corresponding dynamical systems.

    This is considered to be an implementation detail, and should not be useful
    from a user perspective.

    """

    default_name = 'Phase plane'
    """Default name given to instances."""

    def __init__(self, system, axes, fixed_axes, name=None):
        if not isinstance(system, systems.DynamicalSystem):
            raise TypeError('The system associated with a \'PhasePlane\' '
                            'instance must be a \'DynamicalSystem\' instance.')
        self._system = system
        super().__init__(system.config, name=name)
        self.axes = axes
        if (n := self.system.dim - 2) == 0:
            self.fixed_axes = None
        else:
            self.fixed_axes = fixed_axes

    def __new__(cls, system, axes, fixed_axes, name=None):
        if isinstance(system, systems.WilsonCowanSystem):
            return super().__new__(_PhasePlaneWilsonCowan)
        if isinstance(system, (systems.MeanFieldSystem, systems.MixedSystem)):
            return super().__new__(_PhasePlaneMeanField)
        return super().__new__(cls)

    @property
    def system(self):
        """Dynamical system for which to draw a phase plane.

        Dynamical system for which a phase plane is to be drawn. It must be
        a `popnet.systems.DynamicalSystem` instance. It is set at
        initialization, and afterwards it cannot be manually set nor deleted.
        """
        return self._system

    @property
    def axes(self):
        """Components chosen as independant variables for the phase plane.
        
        It must be a tuple of two integers corresponding to valid axes of the
        dynamical system. It cannot be deleted.
        """
        return self._axes

    @axes.setter
    def axes(self, new_value):
        self._check_if_tuple_of_two_int(new_value)
        if max(new_value) >= self.system.dim:
            if (L := self.system.dim) == 1:
                s = ''
            else:
                s = 's'
            raise ValueError(f'{new_value} contains an invalid index: the '
                             'dynamical system has only '
                             f'{self.system.dim} component{s}.')
        self._axes = new_value

    @property
    def fixed_axes(self):
        """Fixed values for remaining state components.

        Values given to state components other than those chosen as
        independent. It is a one-dimensional array, or `None` if the whole
        dynamical system is two-dimensional. 

        It can be set as a single float value, in which case this value is
        given to all fixed axes. It cannot be deleted.
        """
        return self._fixed_axes

    @fixed_axes.setter
    def fixed_axes(self, new_value):
        if new_value is None:
            if (n := self.system.dim - 2) > 0:
                warn('The fixed axes should be given values, since the system '
                     'has more dimensions than two. A zero array has been set.',
                     category=PopNetWarning, stacklevel=2)
                self._fixed_axes = np.zeros(n)
            else:
                self._fixed_axes = None
            return
        try:
            new_value = np.array(new_value, float)
        except Exception:
            raise TypeError('The fixed components must be given as an array '
                            'of floats.')
        if (n := self.system.dim - 2) > 0:
            if new_value.shape == ():
                new_value = new_value * np.ones(n)
            elif new_value.shape != (n,):
                raise ValueError('The given fixed components array has shape '
                                 f'{new_value.shape} but it must have shape '
                                 f'{(n,)}.')
            self._fixed_axes = new_value
        else:
            warn('No state components should be fixed as the system does not '
                 'have more than two state components in total. The \'fixed_'
                 'state\' attribute has been set to \'None\'.',
                 category=PopNetWarning, stacklevel=2)
            self._fixed_axes = None

    def legend(self, lw=2, fontsize=10, ncol=1, handletextpad=0.5, 
               framealpha=1, **kwargs):
        """Generate a legend for the figure.

        Generate a legend for the figure with default options. Same as the base
        class method `Graphics.legend`, but changes the defaults for `ncol` and
        `framealpha` to 1.
        """
        return super().legend(lw=lw, fontsize=fontsize, ncol=ncol, 
                              handletextpad=handletextpad,
                              framealpha=framealpha, **kwargs)

    def plot_nullclines(self, which='both', num=1000, xcolor=None, ycolor=None,
                        xlim=(0,1), ylim=(0,1), mask=False, **kwargs):
        """Plot nullclines on the phase plane.

        Plot nullclines for independent variables on the phase plane. This
        method is mostly intended to be used with the Wilson--Cowan system,
        with its extension with refractory state, or with the 'mixed' system
        that makes a transition between the first two. Indeed, with these
        systems nullclines can be computed easily.

        For dynamical systems that include covariances, nullclines are computed
        numerically using [`root`](https://31c8.short.gy/scipy-optimize-root)
        from SciPy's `optimize` module. However, this feature is still
        experimental, and could lead to unaccurate or unexpected results. A
        warning is issued when this method is used in such a case.

        Parameters
        ----------
        which : {'x', 'y', 'both'}, optional
            Which nullcline to plot. Defaults to `'both'`.
        num : int, optional
            Number of points on nullclines. Defaults to 1000.
        xcolor, ycolor : string or tuple, optional
            Colors given to nullclines. `zcolor` is the color for the nullcline
            where the component drawn on axis *z* does not vary. They must be
            valid Matplotlib colors. Both default to `None`, in which case
            default colors are used.
        xlim, ylim : tuple, optional
            Limits of the horizontal and vertical axis, respectively. Both
            default to (0,1).
        mask : bool, optional
            If `True`, the nullclines are masked on the domain where the sum of
            the two variables is greater than 1. Defaults to `False`.
        **kwargs
            Keyword arguments to be passed to the
            [`plot`](https://31c8.short.gy/ax-plot) method of `PhasePlane.ax`,
            which actually plots the nullclines.

        Warns
        -----
        popnet.exceptions.PopNetWarning
            If nullclines are computed numerically with the experimental
            algorithm.
        """
        if which not in (valid_values := ('x', 'y', 'both')):
            warn(f'Unexpected value {which} for \'which\'. Valid values are '
                 f'{valid_values}. No nullclines will be plotted.',
                 stacklevel=2, category=PopNetWarning)
        plot = {'x': True if which in ('x', 'both') else False,
                'y': True if which in ('y', 'both') else False}
        def mask_coords(x, y):
            indices = x + y <= 1
            return x[indices], y[indices]
        x = np.linspace(xlim[0], xlim[1], num)
        y = np.linspace(ylim[0], ylim[1], num)
        if plot['x']:
            x1, y1 = self._nullcline(x, y, self.axes[0], self.axes[1])
            if mask:
                x1, y1 = mask_coords(x1, y1)
            self.ax.plot(x1, y1, color=xcolor,
                         label=self._nullcline_label(self.axes[0]), **kwargs)
        if plot['y']:
            y2, x2 = self._nullcline(y, x, self.axes[1], self.axes[0])
            if mask:
                x2, y2 = mask_coords(x2, y2)
            self.ax.plot(x2, y2, color=ycolor,
                         label=self._nullcline_label(self.axes[1]), **kwargs)

    def plot_solution(self, **kwargs):
        """Plot a solution of the dynamical system on the phase plane.
        
        Run a numerical integration, and plot the resulting solution on the
        phase plane.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the
            [`plot`](https://31c8.short.gy/ax-plot) method of `PhasePlane.ax`,
            which actually plots the solution.
        """
        from .executors import get_integrator
        integrator = get_integrator(self.system)
        solution = self._run_experiment(integrator, 'ode')
        label = kwargs.pop('label', 'Solution')
        self.ax.plot(solution[0], solution[1], label=label, **kwargs)

    def plot_trajectory(self, act='step', **kwargs):
        """Plot a trajectory of a stochastic process on the phase plane.

        Run a simulation of a stochastic process, and plot the resulting
        trajectory on the phase plane. The configuration used with the phase
        plane must have a defined microscopic structure.

        Parameters
        ----------
        act : {'step', 'sigmoid'}, optional
            Shape of neurons' activation rates for the simulation. If `'step'`,
            a neuron's activation rate is a step function going from zero to
            `popnet.structures.MicroNetwork.alpha` at its threshold
            `popnet.structures.MicroNetwork.theta`. If `'sigmoid'`, a neuron's
            activation rate is the logistic function
            `popnet.structures.Population.F` of the population to which it
            belongs. Defaults to `'step'`.
        **kwargs
            Keyword arguments passed to the
            [`plot`](https://31c8.short.gy/ax-plot) method of `PhasePlane.ax`,
            which actually plots the trajectory.
        """
        from .executors import get_simulator
        simulator = get_simulator(self.config, act=act, mode='individual')
        trajectory = self._run_experiment(simulator)
        label = kwargs.pop('label', 'Trajectory')
        self.ax.plot(trajectory[0], trajectory[1], label=label, **kwargs)

    def quiver(self, shape, xlim=(0,1), ylim=(0,1), **kwargs):
        """Draw the vector field on the initialized figure.

        Draw the vector field on the figure `PhasePlane.fig`. The field is
        plotted as a 2D field of arrows with
        [`matplotlib.axes.Axes.quiver`](https://31c8.short.gy/ax-quiver)
        on the axes `PhasePlane.ax`.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid on which to plot the vector field.
        xlim, ylim : tuple, optional
            Limits of the horizontal and vertical axis, respectively. Both
            default to (0,1).
        **kwargs
            Keyword arguments passed to
            [`matplotlib.axes.Axes.quiver`](https://31c8.short.gy/ax-quiver)
        """
        self._check_if_activated()
        X, Y, dX, dY = self._get_arrows(xlim, ylim, shape)
        self.ax.quiver(X, Y, dX, dY, **kwargs)

    def setup(self, xlim=(0,1), ylim=(0,1), set_xlabel=True, set_ylabel=True,
              fontsize=10, aspect='auto'):
        """Setup the figure.

        Setup the figure `PhasePlane.fig`. Allows to set limits to both axes,
        to choose a font size for labels, and to set the aspect ratio of the
        axes.

        Parameters
        ----------
        xlim, ylim : tuple, optional
            Limits of the horizontal and vertical axis, respectively. Both
            default to (0,1).
        set_xlabel, set_ylabel : bool, optional
            Decide if the axes are labelled. Both default to `True`.
        fontsize : float, optional
            Fontsize of the axes' labels. Defaults to 10.
        aspect : {'auto', 'equal'} or float, optional
            Aspect ratio of the axis scaling. If `'auto'`, the plot fills
            the available area, if `'equal'`, the scaling is the same for
            both axes, and if a float, a square would be stretched such that
            its height is `aspect` times its width. Defaults to `'auto'`.
        """
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        if set_xlabel:
            self.ax.set_xlabel(self._label(self.axes[0]), fontsize=fontsize)
        if set_ylabel:
            self.ax.set_ylabel(self._label(self.axes[1]), fontsize=fontsize, 
                               rotation=0)
        self.ax.axes.set_aspect(aspect)

    def streamplot(self, shape, xlim=(0,1), ylim=(0,1), colorbar=False, 
                   cmap='bone', color=None, mask=False, density=1.5, **kwargs):
        """Draw the vector field on the initialized figure.

        Draw the vector field on the figure `PhasePlane.fig`. The field is
        plotted as streamlines with
        [`matplotlib.axes.Axes.streamplot`](https://31c8.short.gy/ax-streamplot)
        on the axes `PhasePlane.ax`.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid on which to plot the vector field.
        xlim, ylim : tuple, optional
            Limits of the horizontal and vertical axis, respectively. Both
            default to (0,1).
        colorbar : bool, optional
            If `True`, a colorbar is added on the phase plane, where the color
            represents the euclidean norm of the derivative. Defaults to
            `False`.
        cmap : str, optional
            Colormap for the vector field when `colorbar` is `True`. See
            [this page](https://31c8.short.gy/mpl-colormap)
            of Matplotlib's documentation for a list of accepted values.
            Defaults to `'bone'`.
        color : str or tuple, optional
            Color of the vector field when `colorbar` is `False`. It must be a
            valid Matplotlib color. Defaults to `None`, in which case a default
            color is used.
        mask : bool, optional
            If `True`, the vector field is masked (in the sense that it is
            plotted in white) on the domain where the sum of the two variables
            is greater then 1. Defaults to `False`.
        density : float, optional
            Density of the stream lines in the plot. Defaults to 1.5.
        **kwargs
            Keyword arguments passed to [`matplotlib.axes.Axes.streamplot`](
            https://31c8.short.gy/ax-streamplot)
        """
        self._check_if_activated()
        X, Y, dX, dY = self._get_arrows(xlim, ylim, shape)
        if colorbar:
            color = np.sqrt(dX**2 + dY**2)
        if mask:
            to_mask = X + Y > 1
            if colorbar:
                cmap = mpl.cm.get_cmap(cmap, 256)
                new_colors = cmap(np.linspace(0, 1, 256))
                new_colors[0] = np.array([1, 1, 1, 1])
                cmap = mpl.colors.ListedColormap(new_colors)
                color = np.where(to_mask, 0, color)
            else:
                cmap = mpl.colors.ListedColormap(['white', color])
                color = 1 - to_mask
        strm = self.ax.streamplot(X, Y, dX, dY, color=color, cmap=cmap, 
                                  density=density, **kwargs)
        if colorbar:
            self.fig.colorbar(strm.lines, ax=self.ax)

    @staticmethod
    def _check_if_tuple_of_two_int(test_value):
        """Check if `test_value` is a tuple of two int."""
        msg = f'{test_value} is not a tuple of non negative integers.'
        if not isinstance(test_value, tuple):
            raise TypeError(msg)
        if len(test_value) != 2:
            raise ValueError(msg)
        if not all(isinstance(val, int) for val in test_value):
            raise ValueError(msg)
        if not all(val >= 0 for val in test_value):
            raise ValueError(msg)

    def _get_arrows(self, xlim, ylim, shape):
        """Get the arrows to plot the vector field."""
        x = np.linspace(xlim[0], xlim[1], shape[0])
        y = np.linspace(ylim[0], ylim[1], shape[1])
        X, Y = np.meshgrid(x, y)
        dX, dY = self._vector_field(X, Y, shape)
        return X, Y, dX, dY

    def _label(self, j, derivative=False):
        """Get the label associated with the *j*th state component."""
        p = len(self.config.network.populations)
        if p == 1:
            return f'${self._label_one(j, derivative)}$'
        return f'${self._label_many(j, p, derivative)}$'

    def _label_many(self, j, p, derivative):
        """Get the *j*th component's label for *p* populations."""
        dot = '\\dot' if derivative else ''
        if 0 <= j < 2*p:
            m = j % p
            X = 'A' if m == j else 'R'
            J = self.config.network.populations[m].ID
            return f'{dot}{{\\mathcal{{{X}}}}}_{{{J}}}'
        triangle_indices = [(m,n) for m in range(p) for n in range(m,p)]
        if 2*p <= j < round(2*p + p*(p+1)/2):
            m, n = triangle_indices[j - 2*p]
            X = Y = 'A'
        elif round(2*p + p*(p+1)/2) <= j < 2*p + p*(p+1):
            m, n = triangle_indices[j - round(2*p + p*(p+1)/2)]
            X = Y = 'R'
        elif 2*p + p*(p+1) <= j < p*(2*p+3):
            m, n = np.ndindex((p,p))[j - 2*p - p*(p+1)]
            X = 'A'
            Y = 'R'
        J = self.config.network.populations[m].ID
        K = self.config.network.populations[n].ID
        return f'{dot}{{\\mathrm{{C}}}}_{{{X}{Y}}}^{{{J}{K}}}'

    def _label_one(self, j, derivative):
        """Get the *j*th component's label for one population."""
        dot = '\\dot' if derivative else ''
        if j == 0:
            return f'{dot}{{\\mathcal{{A}}}}'
        elif j == 1:
            return f'{dot}{{\\mathcal{{R}}}}'
        elif j == 2:
            return f'{dot}{{\\mathrm{{C}}}}_{{AA}}'
        elif j == 3:
            return f'{dot}{{\\mathrm{{C}}}}_{{RR}}'
        elif j == 4:
            return f'{dot}{{\\mathrm{{C}}}}_{{AR}}'

    def _nullcline(self, Z, W, j, k):
        """Get *j*-nullcline with `Z` and `W` as axes `j` and `k`."""
        warn('For this dynamical system, the algorithm that determines the '
             'nullclines is still experimental. The results might not be '
             'accurate.', category=PopNetWarning, stacklevel=3)
        others = np.arange(self.system.dim)
        others = np.setdiff1d(others, np.array(self.axes))
        state_0 = np.zeros(self.system.dim)
        state_0[others] = self.fixed_axes
        null = {j: [], k: []}
        successes = {j: 0, k: 0}
        U = {j: Z, k: W}
        for m in [j,k]:
            n = k if m == j else j
            for u in U[m]:
                state = state_0.copy()
                state[m] = u
                def f(v):
                    state[n] = v
                    return self.system.vector_field(state)[j]
                opt = root(f,0)
                successes[m] += int(opt.success)
                null[m].append(opt.x)
        if successes[j] / len(Z) > successes[k] / len(W):
            return Z, np.array(null[j])
        return np.array(null[k]), W

    def _nullcline_label(self, j):
        """Label for the nullcline of the axis *j*."""
        p = len(self.config.network.populations)
        if p == 1:
            return f'${self._label_one(j, True)} = 0$'
        return f'${self._label_many(j, p, True)} = 0$'

    def _run_experiment(self, executor, *args, **kwargs):
        """Run an experiment and get the output on `self.axes` versus time."""
        executor.run(*args, **kwargs)
        return executor.states[:,self.axes].transpose()

    def _vector_field(self, X, Y, shape):
        """Vector field to plot.

        Vector field to plot to draw the phase plane, as a function of the
        independant variables arranged as a grid.
        """
        state_list = []
        indep_to_app = X
        k = 0
        for j in range(n := self.system.dim):
            if j in self.axes:
                state_list.append(indep_to_app)
                indep_to_app = Y
                k += 1
            else:
                state_list.append(self.fixed_axes[j-k] * np.ones(shape))
        states = np.array(state_list, float)
        try:
            field = self.system.vector_field(states)
        except Exception:
            field = np.zeros((n,) + shape)
            for j, k in np.ndindex(shape):
                field[:,j,k] = self.system.vector_field(states[:,j,k])
        return field[self.axes[0]], field[self.axes[1]]


class _PhasePlaneWilsonCowan(PhasePlane):
    """Specializes `PhasePlane` for the Wilson--Cowan system.

    Specializes `PhasePlane` for the classical Wilson--Cowan dynamical system.
    All changes to the base class are implementation details.

    """

    def _nullcline(self, Z, W, j, k):
        """Get *j*-nullcline with `Z` and `W` as axes `j` and `k`."""
        popJ = self.config.network.populations[j]
        c = self.config.network.c
        B_diff = np.zeros(Z.shape)
        for L in range(len(self.config.network.populations)):
            if L == k:
                continue
            A = Z if L == j else self.fixed_axes[L - int(L > j) - int(L > k)]
            B_diff += c[j,L] * A
        arg = popJ.beta / popJ.alpha * Z / (1 - (1 + popJ.beta/popJ.gamma) * Z)
        Finv_value = popJ.Finv(arg)
        return Z, 1 / c[j,k] * (Finv_value - self.config.Q[j] - B_diff)


class _PhasePlaneMeanField(PhasePlane):
    """Specializes `PhasePlane` for the simple system.

    Specializes `PhasePlane` for the Wilson--Cowan dynamical system with
    refractory state. All changes to the base class are implementation details.

    """

    def _nullcline(self, Z, W, j, k):
        """Get *j*-nullcline with `Z` and `W` as axes `j` and `k`."""
        p = len(self.config.network.populations)
        if j < p:
            if k < p or k == j + p:
                return self._nullcline_A_analytical(Z, W, j, k, p)
            return self._nullcline_A_numerical(Z, W, j, k, p)
        return self._nullcline_R(Z, W, j, k, p)

    def _nullcline_R(self, Z, W, j, k, p):
        """Get the nullcline for an R component."""
        popJ = self.config.network.populations[j % p]
        A = W if k == j-p else self.fixed_axes[j-p - int(j>k)] * np.ones(W.shape)
        return popJ.beta / popJ.gamma * A, W

    def _nullcline_A_analytical(self, Z, W, j, k, p):
        """Get analytically the nullcline for an A component."""
        popJ = self.config.network.populations[j]
        c = self.config.network.c
        A = [Z if n == j else None if n == k else 
             self.fixed_axes[n - int(n>j) - int(n>k)] for n in range(p)]
        B = np.ones(Z.shape) * self.config.Q[j]
        for L in range(p):
            if L == k:
                continue
            B += c[j,L] * A[L]
        if k < p:
            SJ = 1 - self.fixed_axes[p + j - 2] - A[j]
            with np.errstate(divide='ignore'):
                Finv_value = popJ.Finv(popJ.beta * A[j] / (popJ.alpha * SJ))
            return Z, 1 / c[j,k] * (Finv_value - B)
        if k == j + p:
            return Z, 1 - (1 + popJ.beta / (popJ.alpha * popJ.F(B))) * Z

    def _nullcline_A_numerical(self, Z, W, j, k, p):
        """Get numerically the nullcline for an A component."""
        popJ = self.config.network.populations[j]
        R = self.fixed_axes[p + j - 1 - int(p+j>k)]
        def f(a):
            A = [a if n == j else
                 self.fixed_axes[n - int(n>j)] for n in range(p)]
            B = sum(self.config.network.c[j,L] * A[L] for L in range(p))
            return - popJ.beta * a + popJ.alpha * popJ.F(B) * (1 - a - R)
        a = root(f,0).x
        return a * np.ones(W.shape), W


class Result(Graphics):
    """Results generated using PopNet executors.

    The purpose of `Result` is to handle easily the outputs of numerical
    experiments performed by PopNet. The class `Result` has several methods,
    listed in the [Methods](#result-methods) section below, to create and setup
    a [Matplotlib](https://matplotlib.org/) figure with predefined formatting,
    allowing to easily produce many figures in a consistent format. 

    Although some features would be available with the `Result` class alone,
    it is not inteded to be used by itself, but rather through its subclasses
    `Solution`, `ExtendedSolution`, `Trajectory`, `Statistics` and `Spectrum`.
    Each one of these subclasses implements other features specific to a given
    result case.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration used to obtain the result.
    states : array_like
        State of the network with respect to time.
    times : array_like
        Time.
    name : str, optional
        A name associated with the result. Defaults to `None`, in which case it
        is replaced with `Result.default_name`.

    Attributes
    ----------
    config : popnet.structures.Configuration
        Configuration used to obtain the result. See `Result.config`.
    name : str
        Name associated with the result. See `Result.name`.
    fig : matplotlib.figure.Figure
        A Matplotlib figure. See `Result.fig`.
    ax : matplotlib.axes.Axes
        Axes of `fig`. See `Result.ax`.
    times : array_like
        Time.
    colors : dict
        Colors for each state variable associated with the result. See
        `Result.colors`.
    plot : dict
        Plotting methods for each state variable associated with the result.
        See `Result.plot`.
    A, R, S : array_like
        Vectors of state variables with respect to time.
    CAA, CRR, CSS, CAR, CAS, CRS : array_like
        Covariance matrices between state variables with respect to time, or
        `None` if no such covariances are defined for a `Result` subclass.

    Methods {#result-methods}
    -------
     - `Result.activate` :
        Activate a Matplotlib figure.
     - `Result.legend` :
        Make a legend for the figure.
     - `Result.setup` :
        Setup the axes of the figure.
     - `Result.draw` :
        Draw the figure and show it or save it.

    Raises
    ------
    TypeError
        If `config` is not a `popnet.structures.Configuration` instance.

    Notes
    -----
    Some additional remarks should be made regarding the implementation, in
    case `Result` would have to be subclassed. Everything discussed here is
    considered to be implementation details, and should not be useful from a
    user perspective.

    The class `Result` itself is in fact intended to handle directly only the
    case where the network used in the configuration has more than one
    population. The one population case is internally handled by a private
    subclass `_ResultOne`, which is responsible to modify the state components
    attributes and the methods of the `plot` dictionary, to change them from
    one-element lists to the elements themselves. The same pattern is followed
    in `Result` subclasses: each one has a private subclass with the same name
    but with a suffix 'One'.

    When creating a new `Result` instance, the constructor internally checks
    the number of populations of the network, and if this number is one, then
    the constructor looks in the `graphics` module for a private class of the
    same name suffixed with 'One', and rather instantiates this class if it
    exists.

    """

    default_name = 'Result'
    """Default name given to instances."""
    x_units = 'Time'
    """Units of the horizontal axis. The default is `'Time'`."""
    _lim_valid_values = {'x': ('time', 'config', 'unbounded'), 
                         'y': ('fractions', 'covariances', 'unbounded')}

    def __init__(self, config, states, times, name=None):
        super().__init__(config, name=name)
        self._init_abscissa(times)
        self._init_states_dict(states)
        self._init_colors()
        self._init_plot_methods()

    def __new__(cls, config, states, times, name=None):
        if not isinstance(config, structures.Configuration):
            raise TypeError('The configuration used with a \'Result\' '
                            'instance must be a \'Configuration\' instance.')
        if len(config.network.populations) == 1:
            prefix = '' if cls.__name__.startswith('_') else '_'
            suffix = '' if cls.__name__.endswith('One') else 'One'
            try:
                one_pop_class = globals()[f'{prefix}{cls.__name__}{suffix}']
            except KeyError:
                pass
            else:
                return super().__new__(one_pop_class)
        return super().__new__(cls)

    @classmethod
    def load(cls, ID, name=None, config=None, times=None, folder=None):
        """Load the result associated with the ID.

        Load the result obtained when using the configuration of ID `ID`. The
        array representing the state of the network with respect to time is
        expected to be in a file named *ID - name.txt*, where *ID* and *name*
        are indeed `ID` and `name`. 

        Parameters
        ----------
        ID : str
            ID of the configuration used to obtain this result. 
        name : str, optional
            Name associated with the result. Defaults to `None`, in which case
            it is replaced with the name of the class.
        config : popnet.structures.Configuration, optional
            Configuration to associate with the result. If given, it must have
            the ID `ID`. Defaults to `None`, in which case it is loaded with
            `popnet.structures.load_config`, using the same ID.
        times : array_like, optional
            Times array to associate with the result. Defaults to `None`, in
            which case it is computed from the configuration.
        folder : str, optional
            Folder in which the file is located, which should be placed in the
            current directory. Defaults to `None`, in which case the file is
            assumed to be located in the current directory.

        Returns
        -------
        Result
            The loaded result. 

        Raises
        ------
        TypeError
            If `config` is neither `None` nor a
            `popnet.structures.Configuration` instance.
        popnet.exceptions.PopNetError
            If `config` has a different ID than `ID`. 
        FileNotFoundError
            If no file is found with the expected name.
        """
        config = cls._check_config(config, ID)
        if times is None:
            times = np.linspace(config.initial_time, config.final_time,
                                1 + config.iterations)
        name = cls._get_name(name)
        filename = _internals._format_filename(folder, ID, name)
        try:
            states = np.loadtxt(filename, dtype=float, ndmin=2)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                    f'No result found for configuration {ID}.') from error
        return cls(config, states, times, name)

    @property
    def times(self):
        """Time.

        An array representing time. It is the independant variable with respect
        to which state components are given. It cannot be set nor deleted.
        """
        return self._times

    @property
    def colors(self):
        """Colors associated with the result's components.

        Colors associated with the result's components, to be used in figures.
        It is a dictionary whose keys are strings representing possible
        components, and whose values are lists (or lists of lists) of valid
        Matplotlib colors associated with each population (or pair of
        populations). For example, `result.colors['A'][J]` is the color
        associated with the activity of the *J*th population of the network for
        the result `result`. By default, in the case of a single population,
        a new color can be assigned as is, and it is automatically placed in a
        list (containing only the given color). It cannot be deleted.
        """
        return self._colors

    @colors.setter
    def colors(self, new_colors):
        if not isinstance(new_colors, dict):
            raise TypeError('The colors passed to a \'Result\' instance must '
                            'be stored in a dictionary.')
        self._colors = new_colors

    @property
    def plot(self):
        """Dictionary of methods to plot state variables.

        If *X* denotes a state variable (that is, either *A*, *R* or *S*), then
        `plot['X']` is a list whose *J*th element is a method that plots
        \\(X^J\\) (or its expectation, depending on the
        result) with respect to time. Similarly, if covariances are defined for
        this result, then `plot['CXY'][J][K]` is a method which plots the
        covariances between \\(X^J\\) and \\(Y^K\\) with respect to time. A
        similar pattern works for third central moments as well if applicable,
        with keys of the form `'XYZ'`.

        In the case where the network has only one population, the lists are
        all replaced with the single element they would contain. For example,
        `plot['A']` is directly a method to plot the network's activity, that
        is, the activity of the single population of the network. In the same
        way, `plot['CRR']` is a method to plot the variance of the refractory
        fraction of the network.

        When called, these methods plot the corresponding state components with
        respect to time on the axes `Result.ax` of the figure `Result.fig`. They
        all accept keyword arguments that can be passed to the
        [`plot`](https://31c8.short.gy/ax-plot) method of `Result.ax`. This
        attribute cannot be set nor deleted.
        """
        return self._plot

    @property
    def A(self):
        """Active fractions of populations.

        List of active fractions of populations (or their expectations,
        depending on the result) as arrays indexed with respect to time. If
        the network has only one population, it is not a list, but directly the
        array giving the active fraction of the network with respect to time.

        It cannot be set nor deleted.
        """
        return self._states_dict['A']

    @property
    def R(self):
        """Refractory fractions of populations.

        List of refractory fractions of populations (or their expectations,
        depending on the result) as arrays indexed with respect to time. If
        the network has only one population, it is not a list, but directly the
        array giving the refractory fraction of the network with respect to
        time.

        It cannot be set nor deleted.
        """
        return self._states_dict['R']

    @property
    def S(self):
        """Sensitive fractions of population.

        List of sensitive fractions of populations (or their expectations,
        depending on the result) as arrays indexed with respect to time. If
        the network has only one population, it is not a list, but directly the
        array giving the sensitive fraction of the network with respect to time.

        It cannot be set nor deleted.
        """
        return self._states_dict['S']

    @property
    def CAA(self):
        """Covariances between active fractions of populations.

        Matrix (as a list of lists) of covariances between active fractions of
        populations with respect to time, or `None` if no such matrix is
        defined. If the network has only one population, it is not a list of
        lists, but directly the variance of the active fraction of the network.

        It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CAA']
        except KeyError:
            pass

    @property
    def CRR(self):
        """Covariances between refractory fractions of populations.

        Matrix (as a list of lists) of covariances between refractory fractions
        of populations with respect to time, or `None` if no such matrix is
        defined. If the network has only one population, it is not a list of
        lists, but directly the variance of the refractory fraction of the
        network.

        It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CRR']
        except KeyError:
            pass

    @property
    def CSS(self):
        """Covariances between sensitive fractions of populations.

        Matrix (as a list of lists) of covariances between sensitive fractions
        of populations with respect to time, or `None` if no such matrix is
        defined. If the network has only one population, it is not a list of
        lists, but directly the variance of the sensitive fraction of the
        network.

        It cannot be set nor deleted. 
        """
        try:
            return self._states_dict['CSS']
        except KeyError:
            pass

    @property
    def CAR(self):
        """Covariances between active and refractory fractions of populations.

        Matrix (as a list of lists) of covariances between active and refractory
        fractions of populations with respect to time, or `None` if no such
        matrix is defined. If the network has only one population, it is not a
        list of lists, but directly the covariance between the active and
        refractory fractions of the network.

        It cannot be set nor deleted.
        """
        try:
            return self._states_dict['CAR']
        except KeyError:
            pass

    @property
    def CAS(self):
        """Covariances between active and sensitive fractions of populations.

        Matrix (as a list of lists) of covariances between active and sensitive
        fractions of populations with respect to time, or `None` if no such
        matrix is defined. If the network has only one population, it is not a
        list of lists, but directly the covariance between the active and
        sensitive fractions of the network.

        It cannot be set nor deleted.
        """
        try:
            return self._states_dict['CAS']
        except KeyError:
            pass

    @property
    def CRS(self):
        """Covariances between refractory and sensitive fractions of populations.

        Matrix (as a list of lists) of covariances between refractory and
        sensitive fractions of populations with respect to time, or `None` if no
        such matrix is defined. If the network has only one population, it is
        not a list of lists, but directly the covariance between the refractory
        and sensitive fractions of the network.

        It cannot be set nor deleted.
        """
        try:
            return self._states_dict['CRS']
        except KeyError:
            pass

    def default_figure(self, ncol=None, show=True, savefig=False, **kwargs):
        """Draw a figure with default plots and parameters.

        Draw a figure with default plots and parameters. The curves plotted by
        default are all fractions of populations (or their expectations or
        averages) and their variances, if those are defined for the result. The
        keyword arguments listed in the [Other Parameters](#other-parameters)
        section below give a little more control over the curves which are
        plotted, for results with a lot of components. If the figure is saved,
        it is under *ID - name.png*, where *ID* and *name* are the
        corresponding attributes of the result, and it is placed in the
        current directory.

        Parameters
        ----------
        ncol : int, optional
            Number of columns of the legend, passed to `Graphics.legend`.
            Defaults to `None`, in which case a default value is replaced.
        show : bool
            Decides if the figure is shown or not. Defaults to `True`.
        savefig : bool, optional
            Decides if the figure is saved or not. Defaults to `False`.
        **kwargs
            Keyword arguments to be passed to an internal method that decides
            what is plotted on the figure. Valid keyword arguments are given in
            the [Other Parameters](#other-parameters) section below.
        
        Other Parameters
        ----------------
        expectations : bool, optional
            Decides whether expectations (or averages) are plotted on the
            figure. Valid in `ExtendedSolution`, `Statistics` and `Spectrum`.
            Defaults to `True` whenever it is valid.
        variances : bool, optional
            Decides whether variances are plotted on the figure. Valid in
            `ExtendedSolution`, `Statistics` and `Spectrum`. Defaults to `True`
            whenever it is valid.
        covariances : bool, optional
            Decides whether non-symmetric covariances are plotted on the figure.
            Valid in `ExtendedSolution`, `Statistics` and `Spectrum`. Defaults
            to `False`.
        third_moments : bool, optional
            Decides whether third central moments are plotted on the figure.
            Valid in `Statistics` and `Spectrum`. Defaults to `False`.
        """
        self.activate()
        self._default_plots(**kwargs)
        self.setup()
        self.legend(ncol=ncol)
        self.draw(show=show, savefig=savefig)

    def get_spectrum(self, name=None):
        """Get the spectrum of this result.

        Get a `Spectrum` instance corresponding to `self` where each state
        component is replaced by its real fast Fourier transform.

        Parameters
        ----------
        name : str, optional
            Name to associate with the spectrum. Defaults to `None`, in which
            case it is replaced with `'Spectrum'`.

        Returns
        -------
        Spectrum
            The spectrum of the result `self`.
        """
        return Spectrum(self.config, self._states_dict, self.times, 
                        self.default_name, name)

    def setup(self, set_xlabel=True, units='ms', fontsize=10, xlim='time',
              ylim='fractions'):
        """Setup a figure.

        Setup the figure `Result.fig`. Allows to set an automatic label to the
        horizontal axis based on `Result.x_units`, to choose a font size for
        the labels, and to set limits to both axes.

        Parameters
        ----------
        set_xlabel : bool, optional
            Decides if the horizontal axis is labelled. Defaults to `True`.
        units : str, optional
            Time units for the horizontal axis, indicated in square brackets on
            the figure. If it is set to the empty string `''`, no extra square
            brackets are added. Defaults to `'ms'`, which assumes that
            transition rates are given in kHz.
        fontsize : float, optional
            Font size in points for the horizontal axis' label. Defaults to 10.
        xlim : {'time', 'config', 'unbounded'}, optional
            Decides how the horizontal axis is bounded. If `'time`', it is
            bounded by the initial and final values of the times array. If
            `'config'`, it is bounded by the initial and final times of
            the configuration. If `'unbounded'`, it is not bounded.
            Defaults to `'time'`.
        ylim : {'fractions', 'covariances', 'unbounded'}, optional
            Decides how the vertical axis is bounded. If `'fractions'`, it is
            bounded between 0 and 1. If `'covariances'`, it is bounded between
            between -1/4 and 1. If `'unbounded'`, it is not bounded. Defaults
            to `'fractions'`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Result.fig` and `Result.ax`.
        ValueError
            If `xlim` or `ylim` is given a non-valid value.
        """
        self._check_if_activated()
        self._set_xlabel(set_xlabel, units, fontsize)
        self._check_lim_value('x', xlim)
        self._check_lim_value('y', ylim)
        self._set_xlim(xlim)
        self._set_ylim(ylim)

    def _add_colors_items_one(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to state variables
        involving *one* population, and define some default colors.
        """
        for X in ['A', 'R', 'S']:
            self.colors[X] = [None for pop in self.config.network.populations]
        if (p := len(self.config.network.populations)) == 1:
            self.colors['A'][0] = (150/255,10/255,47/255)
            self.colors['R'][0] = 'midnightblue'
            self.colors['S'][0] = 'goldenrod'
        elif p == 2:
            self.colors['A'] = ['midnightblue', (150/255,10/255,47/255)]
            self.colors['R'] = ['royalblue', 'crimson']
            self.colors['S'] = ['skyblue', (243/255,125/255,148/255)]

    def _add_colors_items_two(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to variables
        involving *two* populations, and define some default colors.
        """
        for XY in ['AA', 'RR', 'SS', 'AR', 'AS', 'RS']:
            key = f'C{XY}'
            self.colors[key] = [[None for p1 in self.config.network.populations]
                                      for p2 in self.config.network.populations]
        if (p := len(self.config.network.populations)) == 1:
            self.colors['CAA'][0][0] = 'salmon'
            self.colors['CRR'][0][0] = 'skyblue'
            self.colors['CSS'][0][0] = 'gold'
            self.colors['CAR'][0][0] = 'violet'
            self.colors['CAS'][0][0] = (255/255,180/255,0)
            self.colors['CRS'][0][0] = 'springgreen'
        elif p == 2:
            self.colors['CAA'] = [['seagreen', None], [None, 'blueviolet']]
            self.colors['CRR'] = [['mediumseagreen', None], 
                                  [None, 'mediumorchid']]
            self.colors['CSS'] = [['springgreen', None], [None, 'violet']]

    def _add_colors_items_three(self):
        """Add items to the `colors` dictionary. 

        Add to the `colors` dictionary items corresponding to variables
        involving *three* populations.
        """
        keys = ['AAA', 'AAR', 'AAS', 'ARR', 'ARS', 
                'ASS', 'RRR', 'RRS', 'RSS', 'SSS']
        for k in keys:
            self.colors[k] = [[[None for p1 in self.config.network.populations]
                                     for p2 in self.config.network.populations]
                                     for p3 in self.config.network.populations]

    @staticmethod
    def _check_config(config, ID):
        """Check if `config` is a valid configuration when loading a result."""
        if config is None:
            return structures.load_config(ID)
        if not isinstance(config, structures.Configuration):
            raise TypeError('The configuration to associate with a loaded '
                            'result must be a \'Configuration\' instance.')
        if config.ID != ID:
            raise PopNetError('The configuration to associate with a result '
                              f'loaded from ID {ID} must have the same ID.')
        return config

    @classmethod
    def _check_lim_value(cls, axis, value):
        """Check if `value` is valid to bound axis `axis`."""
        if value not in cls._lim_valid_values[axis]:
            raise ValueError(f'The value \'{value}\' is not valid for the '
                             f'parameter {axis}lim of setup(). Valid values '
                             f'are {cls._lim_valid_values[axis]}.')

    def _default_plots(self, one=True, symmetric=False, nonsymmetric=False,
                       three=False):
        """Add plots on the default figure."""
        if one: self._plot_all_one()
        self._plot_all_two(symmetric=symmetric, nonsymmetric=nonsymmetric)
        if three: self._plot_all_three()

    def _get_fractions_dict(self, states):
        """Get the fractions variables dictionary."""
        transposed_states = np.transpose(states)
        A = transposed_states[: (p := len(self.config.network.populations))]
        R = transposed_states[p : 2*p]
        S = 1 - A - R
        return {'A': A, 'R': R, 'S': S}

    def _init_abscissa(self, times):
        """Initialize array related to the independant variable."""
        self._times = times

    def _init_colors(self):
        """Initialize the colors associated with state variables."""
        self.colors = _internals.PopNetDict()
        self._add_colors_items_one()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        self._plot = self._plot_dict_one()

    def _init_states_dict(self, states):
        """Initialize the state variables dictionary."""
        self._states_dict = self._get_fractions_dict(states)

    def _label_one(self, X, J):
        pass

    def _label_two(self, CXY, J, K):
        pass

    def _label_three(self, XYZ, J, K, L):
        pass

    def _make_plot_one(self, X, J, states=None, label_func=None, lw=None, ls=None):
        """Define a plotting method for a given state variable.

        Define a method to plot the state variable `X` for the population `J`,
        labeling the curve with `label_func` and taking the data in `states`.
        """
        if states is None:
            states = self._states_dict
        if label_func is None:
            label_func = self._label_one
        def f(add_label=True, lw=lw, ls=ls, color=None, label=None, **kwargs):
            self._check_if_activated()
            if color is None:
                color = self.colors[X][J]
            if add_label and label is None:
                label = label_func(X, J)
            line, = self.ax.plot(self.times, states[X][J], label=label, lw=lw,
                                 ls=ls, color=color, **kwargs)
            return line
        return f

    def _make_plot_two(self, CXY, J, K):
        """Define a plotting method for a covariance.

        Define a method to plot the covariance `CXY` for the `J`th and `K`th
        population.
        """
        def f(color=None, label=None, **kwargs):
            self._check_if_activated()
            if color is None:
                color = self.colors[CXY][J][K]
            if label is None:
                label = self._label_two(CXY[1:], J, K)
            covariance = self._states_dict[CXY][J][K]
            line, = self.ax.plot(self.times, covariance, label=label,
                                 color=color, **kwargs)
            return line
        return f

    def _make_plot_three(self, XYZ, J, K, L):
        """Define a plotting method for a third central moment.
        
        Define a method to plot the third central moment for the state variables
        `X`, `Y` and `Z` for the `J`th, `K`th and `L`th population respectively.
        """
        def f(verbose=True, color=None, label=None, **kwargs):
            self._check_if_activated()
            try:
                moment = self._states_dict[XYZ][J][K][L]
            except (KeyError, IndexError):
                moment = None
            if moment is None:
                if verbose:
                    warn('The third central moment requested to plot is not '
                         'available. If this result has third central moments, '
                         'there should be a transposition of indices that '
                         'allows to get the same moment in another way.',
                         category=PopNetWarning, stacklevel=2)
                return None
            if color is None:
                color = self.colors[XYZ][J][K][L]
            if label is None:
                label = self._label_three(XYZ, J, K, L)
            line, = self.ax.plot(self.times, moment, label=label, 
                                 color=color, **kwargs)
            return line
        return f

    def _plot_all_one(self, **kwargs):
        """Plot all state variables associated with *one* population."""
        for J in range(len(self.config.network.populations)):
            for X in ['A', 'R', 'S']:
                self.plot[X][J](**kwargs)

    def _plot_all_two(self, symmetric=True, nonsymmetric=True, **kwargs):
        """Plot all state variables associated with *two* populations.

        Parameters
        ----------
        symmetric : bool, optional
            Decides whether variances are plotted. Defaults to `True`.
        nonsymmetric : bool, optional
            Decides whether non-symmetric covariances are plotted. Defaults to
            `True`.
        """
        p = len(self.config.network.populations)
        for J, K in np.ndindex((p,p)):
            for CXX in ['CAA', 'CRR', 'CSS']:
                if J == K and symmetric:
                    self.plot[CXX][J][K](**kwargs)
                if J < K and nonsymmetric:
                    self.plot[CXX][J][K](**kwargs)
            for CXY in ['CAR', 'CAS', 'CRS']:
                if nonsymmetric:
                    self.plot[CXY][J][K](**kwargs)

    def _plot_all_three(self, **kwargs):
        """Plot all state variables associated with *three* populations."""
        p = len(self.config.network.populations)
        for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARR',
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']:
            for J, K, L in np.ndindex((p,p,p)):
                self.plot[XYZ][J][K][L](verbose=False, **kwargs)

    def _plot_dict_one(self, **kwargs):
        """Return a dictionary of plotting methods for *one* population."""
        p = len(self.config.network.populations)
        return {X: [self._make_plot_one(X, J, **kwargs) for J in range(p)]
                for X in ['A', 'R', 'S']}

    def _plot_dict_two(self):
        """Return a dictionary of plotting methods for *two* populations."""
        p = len(self.config.network.populations)
        return {CXY: [[self._make_plot_two(CXY, J, K)
                       for K in range(p)] for J in range(p)]
                for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

    def _plot_dict_three(self):
        """Return a dictionary of plotting methods for *three* populations."""
        p = len(self.config.network.populations)
        return {XYZ: [[[self._make_plot_three(XYZ, J, K, L) for L in range(p)]
                        for K in range(p)] for J in range(p)]
                for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARS',
                            'ASS', 'RRR', 'RRS', 'RSS', 'SSS']}

    def _set_xlabel(self, set, units, fontsize):
        """Set the label of the horizontal axis of a figure."""
        if set:
            units = f' [{units}]' if units != '' else ''
            self.ax.set_xlabel(f'{self.x_units}{units}', fontsize=fontsize)

    def _set_xlim(self, xlim):
        """Set the limits of the horizontal axis of a figure."""
        if xlim == 'time':
            self.ax.set_xlim(self.times[0], self.times[-1])
        elif xlim == 'config':
            self.ax.set_xlim(self.config.initial_time, self.config.final_time)
        elif xlim == 'unbounded':
            pass

    def _set_ylim(self, ylim):
        """Set the limits of the vertical axis of a figure."""
        if ylim == 'fractions':
            self.ax.set_ylim([0, 1])
        elif ylim == 'covariances':
            self.ax.set_ylim([-1/4, 1])
        elif ylim == 'unbounded':
            pass


class _ResultOne(Result):
    """Adapts `Result` to the special case of a single population.

    `_ResultOne` adapts `Result` for the case where the network associated with
    the configuration used has only one population.

    The main changes are that the attributes of the form `X` or `CXY` are
    overridden to return directly the corresponding quantity with respect to
    time instead of returning them as lists containing a single element. In the
    same way, the `plot` dictionary values are overridden to be methods rather
    than lists of methods. Also, the default name for a `_ResultOne` class
    trims out the `'One'` suffix of the class' name to leave only `Result`.

    """

    @property
    def A(self):
        """Get the active fraction of the network."""
        return self._states_dict['A'][0]

    @property
    def R(self):
        """Get the refractory fraction of the network."""
        return self._states_dict['R'][0]

    @property
    def S(self):
        """Get the sensitive fraction of the network."""
        return self._states_dict['S'][0]

    @property
    def CAA(self):
        """Get the variance of the active fraction of the network."""
        try:
            return self._states_dict['CAA'][0][0]
        except KeyError:
            pass

    @property
    def CRR(self):
        """Get the variance of the refractory fraction of network."""
        try:
            return self._states_dict['CRR'][0][0]
        except KeyError:
            pass

    @property
    def CSS(self):
        """Get the variance of the sensitive fraction of network."""
        try:
            return self._states_dict['CSS'][0][0]
        except KeyError:
            pass

    @property
    def CAR(self):
        """Covariance between active and refractory fractions of the network."""
        try:
            return self._states_dict['CAR'][0][0]
        except KeyError:
            pass

    @property
    def CAS(self):
        """Covariance between active and sensitive fractions of the network."""
        try:
            return self._states_dict['CAS'][0][0]
        except KeyError:
            pass

    @property
    def CRS(self):
        """Covariance between refractory and sensitive fractions of the network.
        """
        try:
            return self._states_dict['CRS'][0][0]
        except KeyError:
            pass

    def _plot_all_one(self, **kwargs):
        """Plot all state variables associated with *one* population."""
        for X in ['A', 'R', 'S']:
            self.plot[X](**kwargs)

    def _plot_all_two(self, symmetric=True, nonsymmetric=True, **kwargs):
        """Plot all state variables associated with *two* populations."""
        if symmetric:
            for CXX in ['CAA', 'CRR', 'CSS']:
                self.plot[CXX](**kwargs)
        if nonsymmetric:
            for CXY in ['CAR', 'CAS', 'CRS']:
                self.plot[CXY](**kwargs)

    def _plot_all_three(self, **kwargs):
        """Plot all state variables associated with *three* populations."""
        p = len(self.config.network.populations)
        for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARR',
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']:
            self.plot[XYZ](verbose=False, **kwargs)

    def _plot_dict_one(self, **kwargs):
        """Return a dictionary of plotting methods for *one* population."""
        return {X: self._make_plot_one(X, 0, **kwargs) for X in ['A', 'R', 'S']}

    def _plot_dict_two(self):
        """Return a dictionary of plotting methods for *two* populations."""
        return {CXY: self._make_plot_two(CXY, 0, 0)
                for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

    def _plot_dict_three(self):
        """Return a dictionary of plotting methods for *three* populations."""
        return {XYZ: self._make_plot_three(XYZ, 0, 0, 0)
                for XYZ in ['AAA', 'AAR', 'AAS', 'ARR', 'ARS',
                            'ASS', 'RRR', 'RRS', 'RSS', 'SSS']}


class Solution(Result):
    """Represent solutions of dynamical systems.

    `Solution` extends `Result` for the case where the result is a solution
    obtained from a numerical integration of a dynamical system related to the
    Wilson--Cowan model. It adds a method to the base class to plot the
    expectations of fractions of populations all at once. Other changes are
    implementation details.

    """

    default_name = 'Solution'
    """Default name given to instances."""

    def plot_expectations(self, **kwargs):
        """Plot all expectations of fractions of populations.

        Plot all expectations of active, refractory and sensitive fractions of
        populations on the figure `Solution.fig`.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Solution.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Solution.fig` and `Solution.ax`.
        """
        self._plot_all_one(**kwargs)

    def _label_one(self, X, J):
        """Label for the expectation of `X` for the `J`th population."""
        return f'$\\mathcal{{{X}}}_{{{self.config.network.populations[J].ID}}}$'


class _SolutionOne(_ResultOne, Solution):
    """Special case of `Solution` for a single population.

    Special case of `Solution` to use when the network has a single population.
    The only change from the base classes is the implementation of the
    components' labels.

    """

    def _label_one(self, X, J):
        """Label for the expectation of `X`."""
        return f'$\\mathcal{{{X}}}$'


class ExtendedSolution(Solution):
    """Represent solutions of extended dynamical systems.

    `ExtendedSolution` extends `Solution` for cases where covariances are
    considered in the dynamical system that was integrated to obtain the
    solution. It adds methods to the base class to plot variances or
    non-symmetric covariances of fractions of populations all at once. Other
    changes are implementation details.

    """

    default_name = 'Solution (extended)'
    """Default name given to instances."""

    def plot_variances(self, **kwargs):
        """Plot all variances of *A*'s, *R*'s and *S*'s.

        Plot all variances of active, refractory and sensitive fractions of
        populations on the figure `ExtendedSolution.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `ExtendedSolution.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `ExtendedSolution.fig` and
            `ExtendedSolution.ax`.
        """
        self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)

    def plot_covariances(self, **kwargs):
        """Plot all non-symmetric covariances of *A*'s, *R*'s and *S*'s.

        Plot all non-symmetric covariances of active, refractory and sensitive
        fractions of populations on the figure `ExtendedSolution.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `ExtendedSolution.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `ExtendedSolution.fig` and
            `ExtendedSolution.ax`.
        """
        self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)

    def _default_plots(self, expectations=True, variances=True, 
                       covariances=False):
        """Add plots on the default figure."""
        super()._default_plots(one=expectations, symmetric=variances, 
                               nonsymmetric=covariances, three=False)

    def _init_colors(self):
        """Initialize the colors associated with state variables."""
        super()._init_colors()
        self._add_colors_items_two()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        self._plot = {**self.plot, **self._plot_dict_two()}

    def _init_states_dict(self, states):
        """Initialize the attributes associated with state variables."""
        expects = self._get_fractions_dict(states)
        p = len(self.config.network.populations)
        transposed_states = np.transpose(states)
        CAA_flat = transposed_states[2*p : round(2*p + p*(p+1)/2)]
        CAA = _internals._unflat_vector_triangle(CAA_flat)
        CRR_flat = transposed_states[2*p + round(p*(p+1)/2) : 2*p + p*(p+1)]
        CRR = _internals._unflat_vector_triangle(CRR_flat)
        CAR_flat = transposed_states[2*p + p*(p+1) :]
        CAR = CAR_flat.reshape((p,p,len(CAR_flat[0])))
        CAS = - CAA - CAR
        CRS = - CRR - np.transpose(CAR, axes=(1,0,2))
        CSS = - CAS - CRS
        covs = {'CAA': CAA, 'CRR': CRR, 'CSS': CSS,
                'CAR': CAR, 'CAS': CAS, 'CRS': CRS}
        self._states_dict = {**expects, **covs}

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\mathrm{{C}}_{{{XY}}}^{{{IDs}}}$'


class _ExtendedSolutionOne(_SolutionOne, ExtendedSolution):
    """Special case of `ExtendedSolution` for a single population.

    Special case of `ExtendedSolution` to use when the network has a single
    population. The only change from the base classes is the implementation of
    the components' labels.

    """

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y`."""
        return f'$\\mathrm{{C}}_{{{XY}}}$'


class Trajectory(Result):
    """Represent trajectories of stochastic processes.

    `Trajectory` extends `Result` for the case where the result is a possible
    trajectory of a stochastic process that rules the microscopic dynamics
    of the network. It adds a method to the base class to plot the fractions of
    populations all at once, and it adapts the loading method. Other changes
    are implementation details.

    """

    default_name = 'Trajectory'
    """Default name given to instances."""

    @classmethod
    def load(cls, ID, name=None, config=None, folder=None):
        """Load the trajectory associated with the ID.

        Load the trajectory obtained when using the configuration of ID `ID`. It
        extends the base class method by loading the `times` array, which is
        assumed to be in a file named *ID - name (times).txt*, where *ID* and
        *name* are indeed `ID` and `name`. Returns a `Trajectory` instance.

        See Also
        --------
        Result.load
        """
        name = cls._get_name(name)
        filename = _internals._format_filename(folder, ID, f'{name} (times)')
        try:
            times = np.loadtxt(filename, dtype=float)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'No times array found for configuration {ID}') from error
        return super().load(ID, name, config=config, times=times, folder=folder)
        
    def plot_fractions(self, **kwargs):
        """Plot all fractions of populations.

        Plot all active, refractory and sensitive fractions of population on
        the figure `Trajectory.fig`.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Trajectory.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Trajectory.fig` and
            `Trajectory.ax`.
        """
        self._plot_all_one(**kwargs)

    def _label_one(self, X, J):
        """Label for the fraction `X` of the `J`th population."""
        return f'${X}^{{{self.config.network.populations[J].ID}}}$'


class _TrajectoryOne(_ResultOne, Trajectory):
    """Special case of `Trajectory` for a single population.

    Special case of `Trajectory` to use when the network has a single
    population. The only change from the base classes is the implementation of
    the components' labels.

    """

    def _label_one(self, X, J):
        """Label for the fraction `X` of the population."""
        return f'${X}$'


class Statistics(Result):
    """Represent statistics obtained from sample trajectories.

    `Statistics` extends `Result` for the case where the result is a set of
    statistics obtained from multiple trajectories of a stochastic process
    that rules the microscopic evolution of the network.

    The most important extension from the `Result` class is a set of methods to
    plot fills between given bounds around a mean value, and methods to plot the
    minimum and maximum values of a variable at each time step. For details,
    see `Statistics.fill`, `Statistics.plot_max` and `Statistics.plot_min`.
    Besides these new methods, it also adds other methods to plot several state
    components at once, and it adapts the loading method. Other changes are
    implementation details.

    The parameters at initialization are the same as in the base class, except
    that `states` is now expected to be a three-dimensional array of *samples*
    of trajectories of the stochastic process, with time along the first axis,
    state variables along the second, and different simulations along the third.
    Note that this is the format of samples handled by
    `popnet.executors.ChainSimulator`.

    Warns
    -----
    popnet.exceptions.PopNetWarning
        If the given samples do not provide enough trajectories to compute
        unbiased estimates of central moments.

    """

    default_name = 'Statistics'
    """Default name given to instances."""
    default_sample_name = 'Sample'
    """Default name for samples used when loading them."""

    @classmethod
    def load(cls, ID, sample_name=None, name=None, config=None, folder=None):
        """Compute statistics from loaded sample trajectories.

        Compute statistics needed to define a `Statistics` instance from loaded
        samples. For each component *X*, the samples are assumed to be in a
        file named *ID - sample_name X.txt*, where *ID* and *sample_name* are
        indeed `ID` and `sample_name`. In the file for a component *X*, it is
        assumed that in each column are the values of *X* with respect to time
        for a given trajectory.

        Parameters
        ----------
        ID : str
            ID of the configuration used to obtain the samples. 
        sample_name : str, optional
            Name associated with the samples to be loaded. Defaults to `None`,
            in which case is it replaced with `'Sample'`.
        name : str, optional
            Name associated with the result. Defaults to `None`, in which case
            it is replaced with `'Statistics'`.
        config : popnet.structures.Configuration, optional
            Configuration to associate with the result. If given, it must have
            the ID `ID`. Defaults to `None`, in which case it is loaded from ID
            `ID`.
        folder : str, optional
            Folder in which the files are located, which should be placed in the
            current directory. Defaults to `None`, in which case the files are
            assumed to be located in the current directory.

        Returns
        -------
        Statistics
            Statistics computed from the loaded samples.

        Raises
        ------
        TypeError
            If `config` is neither `None` nor a
            `popnet.structures.Configuration` instance.
        popnet.exceptions.PopNetError
            If `config` has a different ID than `ID`. 
        FileNotFoundError
            If no file is found with the expected name for a component.
        """
        config = cls._check_config(config, ID)
        sample_name = cls._get_sample_name(sample_name)
        samples = []
        p = len(config.network.populations)
        for J, X in enumerate(config._variables[:2*p]):
            filename = _internals._format_filename(
                            folder, ID, f'{sample_name} {X}')
            try:
                samples.append(np.loadtxt(filename, dtype=float))
            except FileNotFoundError as e:
                raise FileNotFoundError('No samples found for the component '
                                        f'{X} with configuration {ID}.') from e
        samples = np.transpose(samples, axes=(1,0,2))
        times = np.linspace(config.initial_time, config.final_time, 
                            1 + config.iterations)
        name = cls._get_name(name)
        return cls(config, samples, times, name)

    @property
    def fill(self):
        """Dictionary of methods to add fills for state variables.

        if *X* denotes a state variable (that is, either *A*, *R* or *S*), then
        `fill['X']` is a list whose *J*th element is a method which adds a fill
        around the mean value, between two bounds. By default, the bounds are
        given by one standard deviation on each side of the mean value. These
        methods all accept the same arguments as `Statistics.fill_all`.

        In the case where the network has only one population, the lists are all
        replaced with the single element they would contain. For example,
        `fill['A']` is directly a method to add a fill around the mean value of
        the network's activity.

        When called, these methods add fills for the corresponding state
        components on the axes `Statistics.ax` of the figure `Statistics.fig`.
        They all accept keyword arguments that can be passed to the
        [`fill_between`](https://31c8.short.gy/ax-fill-between) method of
        `Statistics.ax`. This attribute cannot be set nor deleted.
        """
        return self._fill

    @property
    def plot_max(self):
        """Dictionary of methods to plot maxima of state variables.

        Analogous dictionary as `Statistics.plot` with keys `'A'`, `'R'` and
        `'S'`, but were the methods plot *maxima* of state variables instead of
        their mean values.
        """
        return self._plot_max

    @property
    def plot_min(self):
        """Dictionary of methods to plot minima of state variables.

        Analogous dictionary as `Statistics.plot` with keys `'A'`, `'R'` and
        `'S'`, but were the methods plot *minima* of state variables instead of
        their mean values.
        """
        return self._plot_min
        
    def fill_all(self, bound='std', alpha=.25, **kwargs):
        """Add fills between given bounds.

        Add fills between given bounds around the mean values of all fractions
        of populations.

        Parameters
        ----------
        bound : {'std', 'extrema'}, optional
            Describes the bounds between which to fill. If `'std'`, the region
            bounded by one standard deviation around the mean value is filled.
            If `'extrema'`, the region bounded by the minimum and maximum
            values of the component is filled. Defaults to `'std'`.
        alpha : float
            Transparency parameter of the fill. Defaults to 0.25.
        **kwargs
            Keyword arguments to be passed to the method that adds the fills,
            which is the [`fill_between`](https://31c8.short.gy/ax-fill-between)
            method of `Statistics.ax`.
        """
        self._fill_all(bound, alpha, **kwargs)

    def plot_averages(self, **kwargs):
        """Plot all averages of fractions of populations.

        Plot all sample means of active, refractory and sensitive fractions of
        populations on the figure `Statistics.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Statistics.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Statistics.fig` and
            `Statistics.ax`.
        """
        self._plot_all_one(**kwargs)

    def plot_variances(self, **kwargs):
        """Plot all variances of fractions of populations.

        Plot all sample variances of active, refractory and sensitive fractions
        of populations on the figure `Statistics.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Statistics.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Statistics.fig` and
            `Statistics.ax`.
        """
        self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)

    def plot_covariances(self, **kwargs):
        """Plot all non-symmetric covariances of fractions of populations.

        Plot all non-symmetric sample covariances between active, refractory and
        sensitive fractions of populations on the figure `Statistics.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Statistics.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Statistics.fig` and
            `Statistics.ax`.
        """
        self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)

    def plot_third_moments(self, **kwargs):
        """Plot all third central moments of fractions of population.

        Plot all sample third central moments of active, refractory and
        sensitive fractions of populations on the figure `Statistics.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Statistics.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Statistics.fig` and
            `Statistics.ax`.
        """
        self._plot_all_three(**kwargs)

    def _default_plots(self, expectations=True, variances=True,
                       covariances=False, third_moments=False):
        """Add plots on the default figure."""
        super()._default_plots(one=expectations, symmetric=variances, 
                               nonsymmetric=covariances, three=third_moments)

    def _fill_all(self, bound, alpha, **kwargs):
        """Add fills for all *A*'s, *R*'s and *S*'s."""
        p = len(self.config.network.populations)
        for J in range(p):
            for X in ['A', 'R', 'S']:
                self.fill[X][J](bound=bound, alpha=alpha, **kwargs)

    def _fill_dict(self):
        """Return a dictionary of methods to add fills for each population."""
        p = len(self.config.network.populations)
        return {X: [self._make_fill(X, J) for J in range(p)]
                for X in ['A', 'R', 'S']}

    @staticmethod
    def _get_central_moment(samples, stacklevel=1):
        """Compute a central moment from samples.

        Compute a central moment from samples. The axes of `samples` are assumed
        to be ordered in the same way as those handled by `ChainSimulator`, but
        the second axis is assumed to span only the relevant components. Hence,
        if the second axis has length *k*, the central moment computed is of
        order *k*.
        """
        T = len(samples)
        order = len(samples[0])
        executions = len(samples[0,0])
        coeff = 1 / executions
        if order == 2:
            if executions > 1:
                coeff = 1 / (executions - 1)
            else:
                warn('Not enough executions to compute an unbiased estimate of '
                     'a covariance. A biased one is computed instead.',
                     category=PopNetWarning, stacklevel=stacklevel)
        elif order == 3:
            if executions > 2:
                coeff = executions / ((executions - 1) * (executions - 2))
            else:
                warn('Not enough executions to compute an unbiased estimate of '
                     'a third central moment. A biased one is computed '
                     'instead.', category=PopNetWarning, stacklevel=stacklevel)
        means = np.mean(samples, axis=2)
        means = np.resize(means, (executions, T, order)).transpose(1,2,0)
        prod = np.prod(samples - means, axis=1)
        return coeff * np.sum(prod, axis=1)

    @classmethod
    def _get_sample_name(cls, sample_name):
        """Return `sample_name`, or the default sample name if it is `None`."""
        if sample_name is None:
            return cls.default_sample_name
        else:
            return sample_name

    def _init_colors(self):
        """Initialize the colors associated with state variables."""
        super()._init_colors()
        self._add_colors_items_two()
        self._add_colors_items_three()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        self._plot = {**self.plot, **self._plot_dict_two(), 
                      **self._plot_dict_three()}
        self._plot_min = self._plot_dict_one(states=self._min_dict, ls='--',
                                             lw=1, label_func=self._label_min)
        self._plot_max = self._plot_dict_one(states=self._max_dict, ls='--',
                                             lw=1, label_func=self._label_max)
        self._fill = self._fill_dict()

    def _init_states_dict(self, samples):
        """Initialize the state variables dictionary."""
        p = round(len(samples[0]) / 2)
        samples_S = 1 - samples[:,:p] - samples[:,p:]
        samples = np.concatenate((samples, samples_S), axis=1)
        zero = {'A': 0, 'R': p, 'S': 2*p}

        self._min_dict = {X: [np.min(samples[:,zero[X]+J], axis=1) 
                              for J in range(p)]
                          for X in ['A', 'R', 'S']}
        self._max_dict = {X: [np.max(samples[:,zero[X]+J], axis=1) 
                              for J in range(p)]
                          for X in ['A', 'R', 'S']}
        expect = {X: [np.mean(samples[:,zero[X]+J], axis=1) for J in range(p)]
                  for X in ['A', 'R', 'S']}

        def element(CXY, J, K):
            C, X, Y = CXY
            return self._get_central_moment(samples[:,[zero[X]+J, zero[Y]+K]], 
                                            stacklevel=9+int(p==1))

        cov = {CXY: [[element(CXY, J, K) for K in range(p)] for J in range(p)]
              for CXY in ['CAA', 'CRR', 'CSS', 'CAR', 'CAS', 'CRS']}

        def has_to_be_set(X, Y, Z, J, K, L):
            if X == Y == Z:
                return J <= K <= L
            if X == Y:
                return J <= K
            if Y == Z:
                return K <= L
            return True

        def element(XYZ, J, K, L):
            X, Y, Z = XYZ
            if has_to_be_set(X, Y, Z, J, K, L):
                return self._get_central_moment(
                        samples[:,[zero[X]+J, zero[Y]+K, zero[Z]+L]], 
                        stacklevel=10+int(p==1))
            return None

        triplets = ['AAA', 'AAR', 'AAS', 'ARR', 'ARS', 
                    'ASS', 'RRR', 'RRS', 'RSS', 'SSS']
        thirds = {XYZ: [[[element(XYZ, J, K, L) for L in range(p)]
                         for K in range(p)] for J in range(p)]
                  for XYZ in triplets}

        self._states_dict = {**expect, **cov, **thirds}

    def _label_max(self, X, J):
        """Label for the maximum of `X` for the `J`th population."""
        pop = self.config.network.populations[J]
        return f'$\\mathrm{{max}}\\, {{{X}}}_{{{pop.ID}}}$'

    def _label_min(self, X, J):
        """Label for the minimum of `X` for the `J`th population."""
        pop = self.config.network.populations[J]
        return f'$\\mathrm{{min}}\\, {{{X}}}_{{{pop.ID}}}$'

    def _label_one(self, X, J):
        """Label for the expectation of `X` for the `J`th population."""
        return f'$\\mathcal{{{X}}}_{{{self.config.network.populations[J].ID}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\mathrm{{C}}_{{{XY}}}^{{{IDs}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment for variables `X`, `Y` and `Z` for
        the `J`th, `K`th and `L`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K, L)])
        return f'$\\mathrm{{M}}_{{{XYZ}}}^{{{IDs}}}$'

    def _make_fill(self, X, J):
        """Define the method to add a fill around the fraction variable `X` for
        the `J`th population."""
        def f(bound='std', alpha=.25, **kwargs):
            self._check_if_activated()
            if bound == 'std':
                CXX = f'C{X}{X}'
                low = (self._states_dict[X][J] 
                        - np.sqrt(self._states_dict[CXX][J][J]))
                high = (self._states_dict[X][J] 
                        + np.sqrt(self._states_dict[CXX][J][J]))
            elif bound == 'extrema':
                low = self._min_dict[X][J]
                high = self._max_dict[X][J]
            fill = self.ax.fill_between(self.times, low, high, alpha=alpha,
                                        color=self.colors[X][J], **kwargs)
            return fill
        return f


class _StatisticsOne(_ResultOne, Statistics):
    """Special case of `Statistics` for a single population.

    Special case of `Statistics` to use when the network has a single
    population. There are essentially two changes from the base classes:
    the `fill` dictionary values are overridden to be methods rather than lists
    of methods, and components' labels are implemented.

    """
        
    def _fill_all(self, bound, alpha, **kwargs):
        """Add fills for all *A*, *R* and *S*."""
        for X in ['A', 'R', 'S']:
            self.fill[X](bound=bound, alpha=alpha, **kwargs)

    def _fill_dict(self):
        """Return a dictionary of methods to add a fill."""
        return {X: self._make_fill(X, 0) for X in ['A', 'R', 'S']}

    def _label_max(self, X, J):
        """Label for the maximum of `X`."""
        return f'$\\mathrm{{max}}\\, {{{X}}}$'

    def _label_min(self, X, J):
        """Label for the minimum of `X`."""
        return f'$\\mathrm{{min}}\\, {{{X}}}$'

    def _label_one(self, X, J):
        """Label for the expectation of `X`."""
        return f'$\\mathcal{{{X}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y`."""
        return f'$\\mathrm{{C}}_{{{XY}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment between `X`, `Y` and `Z`."""
        return f'$\\mathrm{{M}}_{{{XYZ}}}$'


class Spectrum(Result):
    """Represent spectra of other results.

    `Spectrum` extends `Result` for the case where the result is the spectrum
    of another result. Specifically, it defines methods to plot the spectra of
    state components, and it extends the options to setup a figure. Its data
    attributes are the same as in the base class, but here `times` is replaced
    with `freqs`; see `Spectrum.freqs`. Finally, `Spectrum` forgets
    `Result`'s `load` and `get_spectrum` methods.

    The recommended way of instantiating a `Spectrum` instance is from another
    `Result` instance, with `Result.get_spectrum`. Parameters at initialization
    are a bit different here than in the base class, so they are listed again.

    Parameters
    ----------
    config : popnet.structures.Configuration
        Configuration used to obtain the result.
    states : dict of array_like
        Dictionary in which to each state component is associated with an array.
        Such an array should give the values, for each combination of
        populations, of the state component with respect to time. This is the
        format in which data is kept internally in `Result` classes.
    times : array_like
        An array representing time.
    source : str
        Name of the class from which comes the spectrum.
    name : str, optional
        A name associated with the result. Defaults to `None`, in which case it
        is replaced with `Spectrum.default_name`.

    """

    default_name = 'Spectrum'
    """Default name given to instances."""
    x_units = 'Frequency'
    """Units of the horizontal axis."""
    _lim_valid_values = {'x': ('freqs', 'config', 'unbounded'), 
                         'y': ('unbounded',)}
    _sources_inst = ('Trajectory',)
    _sources_average = ('Solution', 'Solution (extended)', 'Statistics')
    _sources_order_2 = ('Solution (extended)', 'Statistics')
    _sources_order_3 = ('Statistics',)

    def __init__(self, config, states, times, source, name=None):
        if source not in ('Result', 'Solution', 'Solution (extended)',
                          'Trajectory', 'Statistics'):
            raise ValueError(f'Unknown source \'{source}\'.')
        self._source = source
        super().__init__(config, states, times, name)

    def __new__(cls, config, states, times, source, name=None):
        return super().__new__(cls, config, states, times, name=name)

    @classmethod
    def load(cls):
        raise AttributeError('\'Spectrum\' object has no attribute \'load\'')

    @property
    def freqs(self):
        """Frequencies.

        Frequencies for which the Fourier transforms gives the amplitudes.
        Replaces `Result.times`.
        """
        return self.times

    @freqs.setter
    def freqs(self, new_value):
        self.times = new_value

    def get_spectrum(self):
        raise AttributeError('\'Spectrum\' object has no attribute '
                             '\'get_spectrum\'')

    def plot_fractions(self, **kwargs):
        """Plot spectra for all fractions of populations.

        Plot spectra of all active, refractory and sensitive fractions of
        populations on the figure `Spectrum.fig`.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Spectrum.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Spectrum.fig` and `Spectrum.ax`.
        """
        self._plot_all_one(**kwargs)

    def plot_variances(self, **kwargs):
        """Plot spectra of all variances.

        Plot spectra of all variances of active, refractory and sensitive
        fractions of populations on the figure `Spectrum.fig`, if such
        variances are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Spectrum.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Spectrum.fig` and `Spectrum.ax`,
            or if no variances are defined for this result.
        """
        if self._source in self._sources_order_2:
            self._plot_all_two(symmetric=True, nonsymmetric=False, **kwargs)
            return
        raise PopNetError('No variances defined for this result.')

    def plot_covariances(self, **kwargs):
        """Plot spectra of all non-symmetric covariances.

        Plot spectra of all non-symmetric covariances of active, refractory and
        sensitive fractions of populations on the figure `Spectrum.fig`, if
        such covariances are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Spectrum.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Spectrum.fig` and `Spectrum.ax`,
            or if no covariances are defined for this result.
        """
        if self._source in self._sources_order_2:
            self._plot_all_two(symmetric=False, nonsymmetric=True, **kwargs)
            return
        raise PopNetError('No covariances defined for this result.')

    def plot_third_moments(self, **kwargs):
        """Plot spectra of all third central moments.

        Plot spectra of all third central moments of active, refractory and
        sensitive fractions of populations on the figure `Spectrum.fig`, if
        such moments are defined.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to the method that plots components,
            which is the [`plot`](https://31c8.short.gy/ax-plot) method of
            `Spectrum.ax`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Spectrum.fig` and `Spectrum.ax`,
            or if no third central moments are defined for this result.
        """
        if self._source in self._sources_order_3:
            self._plot_all_three(**kwargs)
            return
        raise PopNetError('No third central moments defined for this result.')

    def setup(self, set_xlabel=True, units='kHz', fontsize=10, xlim='freqs',
              yscale='linear'):
        """Setup a figure.

        Setup the figure `Spectrum.fig`. Extends the base class method by
        allowing to set the scale of the vertical axis. Also overrides the
        accepted values for `xlim`, and removes the option to bound the
        vertical axis.

        Parameters
        ----------
        set_xlabel : bool, optional
            Decides if the horizontal axis is labelled. Defaults to `True`.
        units : str, optional
            Frequency units for the horizontal axis, which should be the units
            in which the transition rates are given in the configuration. These
            units are indicated in square brackets on the figure, except if it
            is set to the empty string `''`, in which case no extra square
            brackets are added. Defaults to `'kHz'`.
        fontsize : float, optional
            Font size in points for the horizontal axis' label. Defaults to 10.
        xlim : {'freqs', 'config', 'unbounded'}, optional
            Decides how the horizontal axis is bounded. If `'freqs'`, it is
            bounded between 0 and the highest frequency. If `'config'`, it is
            bounded between 0 and the highest frequency obtained from the times
            array given by the configuration. If `'unbounded'`, it is not
            bounded. Defaults to `'freqs'`.
        yscale : {'linear', 'log'}, optional
            Defines the scale of the vertical axis. Defaults to `'linear'`.

        Raises
        ------
        popnet.exceptions.PopNetError
            If no figure and axes are bound to `Result.fig` and `Result.ax`.
        ValueError
            If `xlim` is given a non-valid value.
        """
        super().setup(set_xlabel=set_xlabel, units=units, fontsize=fontsize,
                      xlim=xlim, ylim='unbounded')
        try:
            self.ax.set_yscale(yscale)
        except AttributeError as error:
            raise TypeError(f'\'yscale\' must be a string.') from error

    def _default_plots(self, expectations=True, variances=True, 
                       covariances=False, third_moments=False):
        """Add plots on the default figure."""
        sym = self._source in self._sources_order_2 and variances
        nonsym = self._source in self._sources_order_2 and covariances
        three = self._source in self._sources_order_3 and third_moments
        super()._default_plots(one=expectations, symmetric=sym, 
                               nonsymmetric=nonsym, three=three)

    def _init_abscissa(self, times):
        """Initialize array related to the independant variable."""
        size = round(len(times)/2+.8)
        # If n is the length of the times array, the size of freqs has to be
        # n/2 + 1 if n is even, and (n+1)/2 if n is odd.
        self._times = np.linspace(0, len(times) / times[-1], size)

    def _init_colors(self):
        """Initialize the colors associated with state variables."""
        super()._init_colors()
        if self._source in self._sources_order_2:
            self._add_colors_items_two()
        if self._source in self._sources_order_3:
            self._add_colors_items_three()

    def _init_plot_methods(self):
        """Initialize plotting methods of state variables."""
        super()._init_plot_methods()
        if self._source in self._sources_order_2:
            self._plot = {**self._plot, **self._plot_dict_two()}
        if self._source in self._sources_order_3:
            self._plot = {**self._plot, **self._plot_dict_three()}

    def _init_states_dict(self, states):
        """Initialize the state variables dictionary."""
        p = len(self.config.network.populations)
        self._states_dict = dict.fromkeys(states)
        axes = {'A': 1, 'R': 1, 'S': 1, 'CAA': 2, 'CRR': 2, 'CSS': 2, 'CAR': 2,
                'CAS': 2, 'CRS': 2, 'AAA': 3, 'AAR': 3, 'AAS': 3, 'ARR': 3,
                'ARS': 3, 'ASS': 3, 'RRR': 3, 'RRS': 3, 'RSS': 3, 'SSS': 3}
        for key in states:
            if key in axes:
                transform = np.fft.rfft(states[key], axis=axes[key])
                self._states_dict[key] = np.abs(transform)
            else:
                raise ValueError(f'{key} is not a valid state variable.')

    def _label_one(self, X, J):
        """Label for the spectrum of `X` for the `J`th population."""
        J_ID = self.config.network.populations[J].ID
        if self._source in self._sources_average:
            return f'$\\hat{{\\mathcal{{{X}}}}}_{{{J_ID}}}$'
        elif self._source in self._sources_inst:
            return f'$\\hat{{{X}}}_{{{J_ID}}}$'

    def _label_two(self, XY, J, K):
        """Label for the spectrum of the covariance between `X` and `Y` for the
        `J`th and `K`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K)])
        return f'$\\hat{{\\mathrm{{C}}}}_{{{XY}}}^{{{IDs}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the spectrum of the third central between `X`, `Y` and `Z`
        for the `J`th, `K`th and `L`th populations."""
        IDs = ''.join([self.config.network.populations[P].ID for P in (J, K, L)])
        return f'$\\hat{{\\mathrm{{M}}}}_{{{XYZ}}}^{{{IDs}}}$'

    def _set_xlim(self, xlim):
        """Set the limits of the horizontal axis of a figure."""
        super()._set_xlim(xlim)
        if xlim == 'freqs':
            self.ax.set_xlim([0, self.freqs[-1]])
        elif xlim == 'config':
            upper = (1+self.config.iterations) / self.config.final_time
            self.ax.set_xlim([0, upper])


class _SpectrumOne(Spectrum, _ResultOne):
    """Special case of `Spectrum` for a single population.

    Special case of `Spectrum` to use when the network has a single population.
    The only change from the base classes is the implementation of the
    components' labels.

    """

    def _label_one(self, X, J):
        """Label for the spectrum of `X`."""
        if self._source in self._sources_average:
            return f'$\\hat{{\\mathcal{{{X}}}}}$'
        elif self._source in self._sources_inst:
            return f'$\\hat{{{X}}}$'

    def _label_two(self, XY, J, K):
        """Label for the covariance between `X` and `Y` for the `J`th and `K`th
        populations."""
        return f'$\\hat{{\\mathrm{{C}}}}_{{{XY}}}$'

    def _label_three(self, XYZ, J, K, L):
        """Label for the third central moment between `X`, `Y` and `Z`."""
        return f'$\\hat{{\\mathrm{{M}}}}_{{{XYZ}}}$'


def draw(name='Figure', show=True, savefig=False, folder=None, format=None,
         **kwargs):
    """Draw a figure.

    Draw a figure previously activated and set up. If the figure is saved, it
    is named `name` and has the file format chosen with `format`.

    Parameters
    ----------
    name : str, optional
        Name to give the figure if saved. Defaults to `'Figure'`.
    show : bool, optional
        Decides if the figure is shown or not. Defaults to `True`.
    savefig : bool, optional
        Decides if the figure is saved or not. Defaults to `False`.
    folder : str, optional
        A folder in which the figure can be saved. If it does not exist in
        the current directory and the figure is saved, it is created. Defaults
        to `None`, in which case the figure is saved in the current directory.
    format : str, optional
        The file format under which the figure is saved if `savefig` is `True`.
        It must be a format handled by Matplotlib, which includes 'png', 'jpg',
        'pdf' and 'svg'. Defaults to `None`, in which case the file format is
        Matplotlib's `savefig.format` parameter, which defaults to 'png'.
    **kwargs
        Keyword arguments passed to [`matplotlib.pyplot.savefig`](
        https://31c8.short.gy/plt-savefig) when `savefig` is `True`.
    """
    if savefig:
        _internals._make_sure_folder_exists(folder)
        filename = _internals._format_filename(folder, None, name, format)
        plt.savefig(filename, format=format, **kwargs)
    if show:
        plt.show()
    plt.close()


def figure(subplots=None, figsize=(5,3.75), dpi=150, tight_layout=True,
           font_family='serif', usetex=False, preamble=None, **kwargs):
    """Initialize a figure.

    Create a Matplotlib figure with default formatting. The default color
    cycle is changed with default colors used in `Graphics` classes. The font
    is also changed to Times or Helvetica according to the font family.

    Parameters
    ----------
    subplots : list or tuple, optional
        If given, one subplot is defined on the figure for each element of
        `subplots`. Each subplot must be specified with an argument understood
        by the [`add_subplot`](https://31c8.short.gy/mpl-add-subplot)
        method of a
        [`matplotlib.figure.Figure`](https://31c8.short.gy/mpl-figure-Figure).
        Defaults to `None`, in which case a single plot is defined.
    figsize : tuple of float, optional
        Width and height of the figure in inches. Defaults to (5, 3.75).
    dpi : int
        Resolution of the figure in dots per inches. Defaults to 150.
    tight_layout : bool, optional
        Adjust automatically the padding between and aroung subplots using
        [`matplotlib.figure.Figure.tight_layout`](
        https://31c8.short.gy/mpl-tight-layout). Defaults to `True`.
    font_family : {'serif', 'sans-serif'}, optional
        Determines if a serif or a sans serif font is used. Defaults to
        `'serif'`.
    usetex : bool, optional
        Determines if LaTeX is used to draw the figure. Defaults to `False`.
    preamble : str, optional
        LaTeX preamble when `usetex` is `True`, in which case it can be used
        to load font packages. It has no effect when `usetex` is `False`.
        Defaults to `None`, in which case a default preamble is added.
    **kwargs
        Keyword arguments to be passed to
        [`matplotlib.pyplot.figure`](https://31c8.short.gy/plt-figure) when
        creating the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The initialized Matplotlib figure.
    ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The axes of `fig`, or a list of axes for every subplot of `fig` if
        multiple subplots have been defined on the figure.

    Warns
    -----
    PopNetWarning
        If an unknown font family is given.
    """
    if font_family not in ['serif', 'sans-serif']:
        warn(f'Unknown font family {font_family}. Taking a default instead.',
             category=PopNetWarning, stacklevel=2)
        font_family = 'serif'
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    if usetex:
        mpl.rcParams['text.usetex'] = True
        if preamble is None:
            if font_family == 'serif':
                preamble = ('\\usepackage{newtxtext}'
                            '\\usepackage{newtxmath}')
            elif font_family == 'sans-serif':
                preamble = ('\\usepackage[cal=pxtx]{mathalpha}'
                            '\\usepackage{helvet}'
                            '\\usepackage{sansmath}'
                            '\\sansmath')
        mpl.rcParams['text.latex.preamble'] = preamble
    else:
        math_font = {'serif': 'stix', 'sans-serif': 'stixsans'}
        mpl.rcParams['mathtext.fontset'] = math_font[font_family]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        'midnightblue', (150/255,10/255,47/255), 'goldenrod', 
        'seagreen', 'blueviolet'])
    fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    if tight_layout:
        fig.set_tight_layout(True)
    if subplots is None:
        subplots = (111,)
    axes = []
    for subplot in subplots:
        if isinstance(subplot, tuple):
            ax = fig.add_subplot(*subplot)
        else:
            ax = fig.add_subplot(subplot)
        ax.tick_params(direction='in', top=True, right=True)
        axes.append(ax)
    if len(axes) == 1:
        return fig, axes[0]
    return fig, axes


def load_extended_solution(ID, name=None, config=None, times=None, folder=None):
    """Load a solution from a text file.

    Load a solution of an extended dynamical system from a text file. This is
    an alias for the method `ExtendedSolution.load` inherited by
    `ExtendedSolution`.
    """
    return ExtendedSolution.load(ID, name=name, config=config, times=times,
                                 folder=folder)


def load_solution(ID, name=None, config=None, times=None, folder=None):
    """Load a solution from a text file.

    Load a solution of a (non extended) dynamical system from a text file.
    This is an alias for the method `Solution.load` inherited by `Solution`.
    """
    return Solution.load(ID, name=name, config=config, times=times,
                         folder=folder)


def load_statistics(ID, sample_name=None, name=None, config=None, folder=None):
    """Alias for `Statistics.load`."""
    return Statistics.load(ID, sample_name=sample_name, name=name,
                           config=config, folder=folder)


def load_trajectory(ID, name=None, config=None, folder=None):
    """Alias for `Trajectory.load`."""
    return Trajectory.load(ID, name=name, config=config, folder=folder)

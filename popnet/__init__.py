"""A python package to study the Wilson--Cowan model and some of its extensions.

PopNet is a python package dedicated to the study of the Wilson--Cowan model
[7], and some of its extensions where the refractory state or covariances
between different fractions of populations are explicitely included. These
extensions are those presented in [4], and throughout this documentation, all
mathematical notation is consistent with the notation used in [4]. PopNet
can perform various numerical experiments, such as integrations of dynamical
systems related to Wilson--Cowan's, or simulations of stochastic processes
whose behavior can be approximated macroscopically by the Wilson--Cowan model.
It has been used to produce all numerical results and figures in [4] and [5].

PopNet provides methods to set, save or load all parameters used for an
expriment, and others to actually perform experiments. It also has a number of
methods to display outputs of experiments. The implementation is heavily
dependent on [NumPy](https://numpy.org/) [2] and
[Matplotlib](https://matplotlib.org/) [3], and uses some functions of
[SciPy](https://scipy.org/) [6] and [`tqdm`](https://tqdm.github.io/) [1]
as well.

The features offered by PopNet are discussed briefly in the [Modules](#modules)
section below, and some conventions regarding expected data structures are
explained in the [Conventions](#conventions) section.


Modules
-------
The package is split into the following four main modules.

 - `popnet.structures` implements several classes to easily group together
   all of the information needed to perform a numerical experiment. It defines
   classes intended to represent biological neural networks and population of
   neurons, as well as configurations which are used to perform experiments.
 - `popnet.systems` implements many dynamical systems related to the 
   Wilson--Cowan model.
 - `executors` implements methods to run numerical experiments, such as
   numerical integrations or simulations of sample trajectories of
   stochastic processes whose behavior can macroscopically be approximated by
   the Wilson--Cowan model.
 - `graphics` implements many classes to handle results of numerical experiments
   performed with the methods of the `popnet.executors` module.

To ease the access to the main features of the package, the functions listed in
the [Functions](#functions) section below, which are all defined in these
modules, are also imported in the package's namespace. Other functions and
classes remain in their module's namespace.


Conventions
-----------
In PopNet, every `popnet.structures.Population`, `popnet.structures.Network`
or `popnet.structures.Configuration` instance is given an ID. These IDs are
meant to ease the task of organizing data. To ensure that there is no confusion
among saved files, some conventions are to be followed:

 - A `popnet.structures.Population`'s ID is always a single character;
 - A `popnet.structures.Network`'s ID always begins with its number of
   populations;
 - A `popnet.structures.Configuration`'s ID always begins with that of its
   network.

Errors are raised when they are not followed.


Fonctions
---------
The following fonctions are all imported in the package's namespace.

  - `popnet.structures.build_network` : Get a network from a weight matrix.
  - `popnet.structures.config` : Define a configuration from a given network.
  - `popnet.structures.default_config` : Define a configuration with default
    parameters.
  - `popnet.structures.default_network` : Define a network with default
    parameters.
  - `popnet.graphics.draw` : Draw a figure.
  - `popnet.graphics.figure` : Initialize a figure with default formatting.
  - `popnet.executors.get_integrator` : Get a numerical integrator.
  - `popnet.executors.get_simulator` : Get a simulator to perform stochastic
    simulations.
  - `popnet.systems.get_system` : Get a dynamical system from a configuration.
  - `popnet.structures.load_config` : Load a configuration from a text file.
  - `popnet.structures.load_network` : Load a network from a text file.
  - `popnet.graphics.load_extended_solution`: Load a solution from a text file.
  - `popnet.graphics.load_solution`: Load a solution from a text file.
  - `popnet.graphics.load_statistics`: Load statistics from text files.
  - `popnet.graphics.load_trajectory`: Load a trajectory from text files.
  - `popnet.structures.network` : Define a network from given populations.
  - `popnet.structures.population` : Define a population of biological neurons.


References
----------
 1. Casper da Costa-Luis, Stephen Karl Larroque, Kyle Altendorf, Hadrien Mary,
    richardsheridan, Mikhail Korobov, Noam Yorav-Raphael, et al. “tqdm: A Fast,
    Extensible Progress Bar for Python and CLI.” Zenodo (2021).
    doi:[10.5281/zenodo.4663456](https://doi.org/10.5281/zenodo.595120).
 2. Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers,
    Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian
    Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H.
    van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark
    Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy,
    Warren Weckesser, Hameer Abbasi, Christoph Gohlke & Travis E. Oliphant.
    “Array programming with NumPy.” *Nature* **585**, 357–362 (2020).
    doi:[10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
 3. John D. Hunter. “Matplotlib: A 2D Graphics Environment.” *Computing in
    Science & Engineering* **9**, 90-95 (2007).
    doi:[10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
 4. Vincent Painchaud. “Dynamique markovienne ternaire cyclique sur graphes et
    quelques applications en biologie mathématique.” Master's thesis, Université
    Laval (2021).
 5. Vincent Painchaud, Nicolas Doyon and Patrick Desrosiers. “Beyond
    Wilson--Cowan dynamics: oscillations and chaos without inhibition.”
    *arXiv:2204.00583* [physics, q-bio] (2022).
    doi:[10.48550/arXiv.2204.00583](https://doi.org/10.48550/arXiv.2204.00583)
 6. Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler
    Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser,
    Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K.
    Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert
    Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake
    VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen,
    E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro,
    Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors.
    “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.”
    *Nature Methods*, **17** (3), 261-272 (2020).
    doi:[10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
 7. Hugh R. Wilson and Jack D. Cowan. “Excitatory and Inhibitory Interactions
    in Localized Populations of Model Neurons.” *Biophysical Journal* **12**
    (1): 1–24 (1972). doi:[10.1016/S0006-3495(72)86068-5](
    https://doi.org/10.1016/S0006-3495(72)86068-5).

"""

from .exceptions import *
from .structures import (build_network, config, default_config, default_network,
                         load_config, load_network, network, population)
from .systems import get_system
from .executors import get_integrator, get_simulator
from .graphics import (figure, draw, load_extended_solution, load_solution, 
                       load_statistics, load_trajectory)

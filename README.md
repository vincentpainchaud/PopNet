# PopNet #

PopNet is a python package dedicated to the study of the Wilson&ndash;Cowan model, some of its extensions, and its underlying stochastic dynamics. It defines classes intended to easily define the parameters of the model, to integrate dynamical systems related to the model or to simulate the microscopic dynamics, and to interpret the solutions.

This package is part of work supervised by Nicolas Doyon and [Patrick Desrosiers](https://github.com/pdesrosiers).

## Requirements ##

- Python 3.8
- SciPy 1.5
- NumPy 1.19
- Matplotlib 3.3

## Installation ##

1. Download the `popnet` folder. 
2. Place it in a directory specified in your `PYTHONPATH` environment variable.
3. Import it in other files and have fun!

## Usage ##

The following example gives the idea of the general usage of the package. 

```python
import popnet as pn


pop = pn.population('Population')
net = pn.network('1A', pop)

config = pn.config(net)
config.initial_state = [.7, .2, .01, .01, .01] #A, R, CAA, CRR, CAR

integrator = pn.get_integrator(config, 'extended')
integrator.run('ode')
solution = integrator.output()
integrator.close()

solution.default_figure()
```
The example outputs the following figure.
![Example output](https://user-images.githubusercontent.com/50606125/131586334-531544eb-357f-490f-821c-d5d63f00332b.png)

## Author ##

Vincent Painchaud

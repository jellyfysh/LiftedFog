## LiftedFog
This repository accompanies the article 
[Lifting the fog - a case for non-reversible "lifted" Markov chains](    
https://doi.org/10.48550/arXiv.2603.16855
),
by Gabriele Tartero, Sora Shiratani, and Werner Krauth.
It notably contains the source code of ```lj-ecmc```, a Rust software package 
used to run the high-precision event-chain Monte Carlo simulations
discussed in the manuscript.

### lj-ecmc
The ```lj-ecmc``` software package, entirely developed by Sora Shiratani, 
is a Rust implementation of the event-chain Monte Carlo (ECMC) algorithm
for the two-dimensional Lennard-Jones potential (without cutoff).
The presence of a suitable cell system makes the algorithm's complexity independent of 
the system size, even in the presence of a truly long-range interaction
(for details on this method, called cell-veto, see the paper in 
[MCLongRange](https://github.com/jellyfysh/MCLongRange.git)).
With a proper choice of the cell-veto parameters, ```lj-ecmc``` is able to compute 
more than $10^9$ events per hour, thus being at least one hundred times faster than any other 
existing ECMC implementation for long-range interacting systems.

This software package
is organized as a standard cargo-based Rust project.
See the [official documentation](
https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html) 
for details.

### Python programs
Several observations and preliminary results concerning the ECMC dynamics
were obtained using [JeLLyFysh](https://github.com/jellyfysh/JeLLyFysh.git) (V1.1),
an open-source Python application for all-atom ECMC simulations.

The factorized Metropolis simulations were performed with an accelerated version of the 
reversible cell-veto algorithm available in 
[MCLongRange](https://github.com/jellyfysh/MCLongRange.git).

### Authors
The authors of this project are:
* Gabriele Tartero 
([gabriele.tartero@phys.ens.fr](mailto:gabriele.tartero@phys.ens.fr));
* Sora Shiratani ([hclfuranponcompotayamano@g.ecc.u-tokyo.ac.jp](mailto:hclfuranponcompotayamano@g.ecc.u-tokyo.ac.jp));
* Werner Krauth ([werner.krauth@ens.fr](mailto:werner.krauth@ens.fr)).

For any question about the ```lj-ecmc``` software package, the Python programs or the related
paper, please raise an issue here on GitHub or contact us via e-mail.

### Additional contributors
The authors are grateful to Kush Patel (University of Oxford) for help in accelerating
the factorized Metropolis simulations with [Numba](https://numba.pydata.org).

### License
This project is licensed under the GNU General Public License, 
version 3 (see the 
[LICENSE](https://github.com/jellyfysh/MCMCNutshell/blob/master/LICENSE) 
file).



# Source and format

The 18 files `08.inp` ... `25.inp` are the butyronitrile C-CN bond-stretch
geometries from the supporting data of

> M. Rossmannek, F. Pavošević, A. Rubio, and I. Tavernelli,
> *Quantum Embedding Method for the Simulation of Strongly Correlated Systems
> on Quantum Computers*, J. Phys. Chem. Lett. **14**, 3491-3497 (2023).
> DOI: [10.1021/acs.jpclett.3c00330](https://doi.org/10.1021/acs.jpclett.3c00330),
> arXiv: [2302.03052](https://arxiv.org/abs/2302.03052).

Each file is plain XYZ text despite the `.inp` extension: a line with `12`,
then 12 `element x y z` rows.  The stem encodes the C-CN distance in Angstrom
(`08` -> 0.8 A, `25` -> 2.5 A).  Atoms 0 and 1 are the N and C of the breaking
bond.

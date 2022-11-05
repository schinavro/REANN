
import sys
from ase.io import read

from pymatgen.core import Structure
from monty.serialization import loadfn

filename = sys.argv[1]

data = loadfn(filename)

pbc_encode = {True: '1', False: '0'}
#tables = {'Cu': 29, 'Ge': 32, 'Li': 3, 'Mo': 42, 'Ni':28, 'Si': 14}
sym_encode = {14: 'Si', 28: 'Ni', 42: 'Mo', 3: 'Li', 32: 'Ge', 29: 'Cu'}
mas_encode = {14: 28.086, 28: 58.693, 42: 95.95, 3: 6.941, 32: 69.723, 29: 63.536}
with open('configuration', 'w') as fd:
    for i, d in enumerate(data):
        fd.write('point=  {0:>4d}'.format(i+1) + '\n') 
        cell = d['structure'].lattice.matrix
        fd.write('  '.join([str(abc) for abc in cell[0]]) + '\n')
        fd.write('  '.join([str(abc) for abc in cell[1]]) + '\n')
        fd.write('  '.join([str(abc) for abc in cell[2]]) + '\n')
        pbc = d['structure'].lattice.pbc
        fd.write('pbc ' + ' '.join([pbc_encode[p] for p in pbc]) + '\n')
        sym = [sym_encode[n] for n in d['structure'].atomic_numbers]
        mas = [mas_encode[n] for n in d['structure'].atomic_numbers]
        positions = d['structure'].cart_coords
        forces = d['outputs']['forces']
        energy = d['outputs']['energy']
        for a in range(len(sym)):
            fd.write('{0:s}  {1:f}  {2:f}  {3:f}  {4:f}  {5:f}  {6:f}  {7:f}\n'.format(
                     sym[a], mas[a], *positions[a], *forces[a])) 
        fd.write('abprop: %f\n' % energy)




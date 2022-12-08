import pybel
import argparse
import utils


def main(filename):
    unit_list = utils.make_unit_list()

    poly = utils.filename_to_poly(filename)
    smiles = utils.make_polymer_smi(poly, unit_list)

    # make polymer string into pybel molecule object
    mol = pybel.readstring('smi', smiles)
    utils.make3D(mol)
    # write polymer .mol file to containing folder
    mol.write('mol', '/ihome/ghutchison/dch45/ga_methods/github/GA-Best-Practices/input_mol/%s.mol' % (filename), overwrite=True)

if __name__ == '__main__':
    usage = "usage: %prog [options] "
    parser = argparse.ArgumentParser(usage)

    # sets input arguments
    # filename
    parser.add_argument('filename', action='store', type=str)

    args = parser.parse_args()

    main(args.filename)
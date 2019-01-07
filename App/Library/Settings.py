import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', action='store_true', default=False, dest='useParameters', help='reset model parameters')

parser.add_argument('-i', action='store', dest='inputFile', help='data input file')
parser.add_argument('-m', action='store', dest='modelPath', help='parameter output path')

parser.add_argument('-n', action='store_true', default=False, dest='newModel', help='reset model parameters')

parser.add_argument('-f', action='store', dest='forexType', help='Kind of forex class')

settings = parser.parse_args()

print("Parameters loaded:", settings)
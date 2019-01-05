import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', action='store', dest='inputPath', help='data input path')
parser.add_argument('-m', action='store', dest='modelPath', help='parameter output path')

parser.add_argument('-n', action='store_true', default=False, dest='newModel', help='reset model parameters')

settings = parser.parse_args()

print("Parameters loaded:", settings)
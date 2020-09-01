#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import h5py
import numpy as np

input_file = h5py.File(sys.argv[1], 'r')
output_file = h5py.File(sys.argv[2], 'w')

fc1 = np.array(input_file['nn']['fc1'])
fc2 = np.array(input_file['nn']['fc2'])
vec = np.array(input_file['nn']['vec'])
y   = np.array(input_file['nn']['y'])

imm = fc1.dot(vec)
relu = np.maximum(0, imm)
res = fc2.dot(relu)
argmax = np.argmax(res)

output_file.create_dataset('argmax', data=argmax)

Edres = res - y
Edfc2 = np.outer(Edres, relu)
Edrelu = fc2.T.dot(Edres)
relu_copy = relu.copy()
relu_copy[relu.nonzero()] = 1
Edimm = Edrelu * relu_copy
Edfc1 = np.abs(np.outer(Edimm, vec))

fc1_index = np.unravel_index(Edfc1.argmax(), Edfc1.shape)
output_file.create_dataset('fc1_max_pos', data=fc1_index)

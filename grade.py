#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, time, os, subprocess, time, shutil, json, h5py
import numpy as np

testcases = [
    ('data/in_1.hdf5', 'data/ans_1.hdf5'),
    ('data/in_2.hdf5', 'data/ans_2.hdf5'),
    ('data/in_3.hdf5', 'data/ans_3.hdf5'),
    ('data/in_4.hdf5', 'data/ans_4.hdf5'),
    ('data/in_5.hdf5', 'data/ans_5.hdf5'),
    ('data/in_6.hdf5', 'data/ans_6.hdf5'),
    ('data/in_7.hdf5', 'data/ans_7.hdf5'),
    ('data/in_8.hdf5', 'data/ans_8.hdf5')
]

if __name__ == '__main__':

    if sys.version_info[0] != 3:
        print("Please use python3")
        exit(1)

    program_file = 'nn.py'
    
    if not os.path.isfile(program_file):
        print('File {} not present!'.format(program_file))
        exit(1)

    success_count = 0
    dump_error = []

    for input, output in testcases:
        # remove the output file
        test_filename = 'grade.hdf5'
        try:
            os.remove(test_filename)
        except:
            pass
        p = subprocess.Popen([sys.executable, program_file, input, test_filename], stdout=open(os.devnull,'w'), stderr=open(os.devnull,'w'))
        message = ''
        success = True
        derivative_suc = True
        start_time = time.time()
        end_time = start_time
        while p.poll() is None:
            if time.time() - start_time > 1:
                p.terminate()
                message = 'Time limit exceeded'
                success = False
                dump_error.append({input: message})
        else:
            if not os.path.isfile(test_filename):
                message = 'No output file found'
                success = False
                dump_error.append({input: message})
            else:
                std = h5py.File(output, 'r')
                try:
                    ans = h5py.File(test_filename, 'r')
                except:
                    success = False
                    break
                if 'argmax' not in ans:
                    message = 'no argmax in output'
                    success = False
                    dump_error.append({input: message})
                elif np.array(std['argmax']) != np.array(ans['argmax']):
                    message = 'argmax expect \'{}\', but get \'{}\''\
                            .format(np.array(std['argmax']), 
                                    np.array(ans['argmax']))
                    success = False
                    dump_error.append({input: message})
                elif 'fc1_max_pos' not in ans:
                    message = 'no fc1_max_pos in output'
                    derivative_suc = False
                    dump_error.append({input: message})
                elif (np.array(std['fc1_max_pos']) != np.array(ans['fc1_max_pos'])).any():
                    message = 'fc1_max_pos expect \'{}\', but get \'{}\''\
                            .format(np.array(std['fc1_max_pos']),
                                    np.array(ans['fc1_max_pos']))
                    derivative_suc = False
                    dump_error.append({input: message})
                else:
                    end_time = time.time()
        if success:
            success_count += 1
            if derivative_suc:
                success_count += 1
                if os.isatty(1):
                    print('Testcase {}: PASS, time {:.3f}s'.format(input,
                        end_time - start_time))
        if not success or not derivative_suc:
            if os.isatty(1):
                print('Testcase {}: {}'.format(input, message))
        
        
    grade = int(100.0 * success_count / (2 * len(testcases)))
    
    if os.isatty(1):
        print('Total Points: {}/100'.format(grade))
    else:
        print(json.dumps({'grade': grade}))

    try:
        os.remove(test_filename)
    except:
        pass

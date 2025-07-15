import argparse
import pickle as pkl
import parse
import sinter

import random
import copy
import math
import ldpc.codes
from ldpc import BpOsdDecoder
import scipy

import numpy as np
import sys
sys.path.append("C:/Users/robin/Desktop/gbstim_pinball")

from gbstim.bposd import BPOSD
from gbstim.gb import CustomUnpickler
from time import time

import re

# def generate_tasks(code, p_range, idle, w, t1, t2):
#     tasks = []
#     for p in p_range:
#         tasks.append(
#             sinter.Task(
#                 circuit=code.stim_circ(
#                     p, p, p, t1, t2, dec_type="Z", idle=idle, num_rounds=code.d
#                 ),
#                 json_metadata={
#                     "p": p,
#                     "Code": f"[{code.n}, {code.k}, {code.d}] Weight {w}",
#                 },
#             )
#         )
#     return tasks

def generate_task(code, p, idle, w, t1, t2):
    this_task = []
    this_task.append(
        sinter.Task(
            circuit=code.stim_circ(
                p, p, p, t1, t2, dec_type="Z", idle=idle, num_rounds=code.d
            ),
            json_metadata={
                "p": p,
                "Code": f"[{code.n}, {code.k}, {code.d}] Weight {w}",
            },
        )
    )
    return this_task

def generate_disjoint_check_classes(m, n, a, b, c, d):
    a0 = m - a
    if a < a0:
        plusrow = a + 1
    else:
        plusrow = a0 + 2
    d0 = n - d
    if d < d0:
        pluscol = d + 1
    else:
        pluscol = d0 + 2
    classes = []
    topL_class = []
    bottomL_class = []
    farL_class = []
    for x0 in range(m):
        for y in range(n):
            x = (x0 * plusrow) % m
            new_topL = [(x % m, y % n), ((x-1) % m, y % n), ((x-1+a) % m, (y+b) % n)]
            overlap = 0
            for existing_topL in topL_class:
                if any(tuple in existing_topL for tuple in new_topL):
                    overlap = 1
            if overlap==0:
                topL_class.append(new_topL)
            new_bottomL = [(x % m, y % n), ((x+1) % m, y % n), ((x+a) % m, (y+b) % n)]
            overlap = 0
            for existing_bottomL in bottomL_class:
                if any(tuple in existing_bottomL for tuple in new_bottomL):
                    overlap = 1
            if overlap==0:
                bottomL_class.append(new_bottomL)
            new_farL = [(x % m, y % n), ((x-a) % m, (y-b) % n), ((x-a+1) % m, (y-b) % n)]
            overlap = 0
            for existing_farL in farL_class:
                if any(tuple in existing_farL for tuple in new_farL):
                    overlap = 1
            if overlap==0:
                farL_class.append(new_farL)
    for x0 in range(m):
        for y in range(n):
            x = (x0 * plusrow + plusrow // 2) % m
            new_topL = [(x % m, y % n), ((x-1) % m, y % n), ((x-1+a) % m, (y+b) % n)]
            overlap = 0
            for existing_topL in topL_class:
                if any(tuple in existing_topL for tuple in new_topL):
                    overlap = 1
            if overlap==0:
                topL_class.append(new_topL)
            new_bottomL = [(x % m, y % n), ((x+1) % m, y % n), ((x+a) % m, (y+b) % n)]
            overlap = 0
            for existing_bottomL in bottomL_class:
                if any(tuple in existing_bottomL for tuple in new_bottomL):
                    overlap = 1
            if overlap==0:
                bottomL_class.append(new_bottomL)
            new_farL = [(x % m, y % n), ((x-a) % m, (y-b) % n), ((x-a+1) % m, (y-b) % n)]
            overlap = 0
            for existing_farL in farL_class:
                if any(tuple in existing_farL for tuple in new_farL):
                    overlap = 1
            if overlap==0:
                farL_class.append(new_farL)
    classes.append(topL_class)
    classes.append(bottomL_class)
    classes.append(farL_class)
    leftR_class = []
    rightR_class = []
    farR_class = []
    for y0 in range(n):
        for x in range(m):
            y = (y0 * pluscol) % n
            new_leftR = [(x % m, y % n), (x % m, (y-1) % n), ((x+c) % m, (y-1+d) % n)]
            overlap = 0
            for existing_leftR in leftR_class:
                if any(tuple in existing_leftR for tuple in new_leftR):
                    overlap = 1
            if overlap==0:
                leftR_class.append(new_leftR)
            new_rightR = [(x % m, y % n), (x % m, (y+1) % n), ((x+c) % m, (y+d) % n)]
            overlap = 0
            for existing_rightR in rightR_class:
                if any(tuple in existing_rightR for tuple in new_rightR):
                    overlap = 1
            if overlap==0:
                rightR_class.append(new_rightR)
            new_farR = [(x % m, y % n), ((x-c) % m, (y-d) % n), ((x-c) % m, (y-d+1) % n)]
            overlap = 0
            for existing_farR in farR_class:
                if any(tuple in existing_farR for tuple in new_farR):
                    overlap = 1
            if overlap==0:
                farR_class.append(new_farR)
    for y0 in range(n):
        for x in range(m):
            y = (y0 * pluscol + pluscol // 2) % n
            new_leftR = [(x % m, y % n), (x % m, (y-1) % n), ((x+c) % m, (y-1+d) % n)]
            overlap = 0
            for existing_leftR in leftR_class:
                if any(tuple in existing_leftR for tuple in new_leftR):
                    overlap = 1
            if overlap==0:
                leftR_class.append(new_leftR)
            new_rightR = [(x % m, y % n), (x % m, (y+1) % n), ((x+c) % m, (y+d) % n)]
            overlap = 0
            for existing_rightR in rightR_class:
                if any(tuple in existing_rightR for tuple in new_rightR):
                    overlap = 1
            if overlap==0:
                rightR_class.append(new_rightR)
            new_farR = [(x % m, y % n), ((x-c) % m, (y-d) % n), ((x-c) % m, (y-d+1) % n)]
            overlap = 0
            for existing_farR in farR_class:
                if any(tuple in existing_farR for tuple in new_farR):
                    overlap = 1
            if overlap==0:
                farR_class.append(new_farR)
    classes.append(leftR_class)
    classes.append(rightR_class)
    classes.append(farR_class)
    return classes

def generate_check_array_array_72(m, n, t_max, stim_syndrome, detector_coordinates):
    check_array_array = [[[0 for _ in range(n)] for _ in range(m)] for _ in range(t_max)]
    for i in stim_syndrome:
        coordinate = detector_coordinates.get(i) 
        x_c, y_c, t_c = coordinate
        x = int((10 - y_c) / 2)
        y = int(((10 - x_c) / 2 + 1) % 6)
        t = int(t_c)
        check_array_array[t][x][y] = 1
    return check_array_array

def generate_decoded_array_array(actual_check_array_array, classes, m, n, a, b, c, d):
    check_array_array = copy.deepcopy(actual_check_array_array)
    correction_array_array = []
    for t in range(len(check_array_array)):
        correction_array = [[0] * n for _ in range(2*m)]
        for cls in range(6):
            num_triangles = len(classes[cls])
            for triangle in range(num_triangles):
                check1_x = classes[cls][triangle][0][0]
                check1_y = classes[cls][triangle][0][1]
                check2_x = classes[cls][triangle][1][0]
                check2_y = classes[cls][triangle][1][1]
                check3_x = classes[cls][triangle][2][0]
                check3_y = classes[cls][triangle][2][1]
                check1 = check_array_array[t][check1_x][check1_y]
                check2 = check_array_array[t][check2_x][check2_y]
                check3 = check_array_array[t][check3_x][check3_y]
                andval = check1 and check2 and check3 
                check_array_array[t][check1_x][check1_y] ^= andval
                check_array_array[t][check2_x][check2_y] ^= andval
                check_array_array[t][check3_x][check3_y] ^= andval
                
                x = check1_x
                y = check1_y

                if cls==0: # top L
                    data_x = (2*x-1) % (2*m)
                    data_y = y % n
                if cls==1: # bottom L
                    data_x = (2*x+1) % (2*m)
                    data_y = y % n
                if cls==2: # far L
                    data_x = (2*(x-a+1)-1) % (2*m)
                    data_y = (y-b) % n
                if cls==3: # left R
                    data_x = (2*x) % (2*m)
                    data_y = (y-1) % n
                if cls==4: # right R
                    data_x = (2*x) % (2*m)
                    data_y = y % n
                if cls==5: # far R
                    data_x = (2*(x-c)) % (2*m)
                    data_y = (y-d) % n
                correction_array[data_x][data_y] ^= andval
        correction_array_array.append(correction_array)
    return check_array_array, correction_array_array

def generate_decoded_data_array(actual_check_array, classes, m, n, a, b, c, d):
    check_array = copy.deepcopy(actual_check_array)
    data_array = [[0] * n for _ in range(2*m)]
    for cls in range(6):
        num_triangles = len(classes[cls])
        for triangle in range(num_triangles):
            check1_x = classes[cls][triangle][0][0]
            check1_y = classes[cls][triangle][0][1]
            check2_x = classes[cls][triangle][1][0]
            check2_y = classes[cls][triangle][1][1]
            check3_x = classes[cls][triangle][2][0]
            check3_y = classes[cls][triangle][2][1]
            check1 = check_array[check1_x][check1_y]
            check2 = check_array[check2_x][check2_y]
            check3 = check_array[check3_x][check3_y]
            andval = check1 and check2 and check3 
            check_array[check1_x][check1_y] ^= andval
            check_array[check2_x][check2_y] ^= andval
            check_array[check3_x][check3_y] ^= andval

            x = check1_x
            y = check1_y

            if cls==0: # top L
                data_x = (2*x-1) % (2*m)
                data_y = y % n
            if cls==1: # bottom L
                data_x = (2*x+1) % (2*m)
                data_y = y % n
            if cls==2: # far L
                data_x = (2*(x-a+1)-1) % (2*m)
                data_y = (y-b) % n
            if cls==3: # left R
                data_x = (2*x) % (2*m)
                data_y = (y-1) % n
            if cls==4: # right R
                data_x = (2*x) % (2*m)
                data_y = y % n
            if cls==5: # far R
                data_x = (2*(x-c)) % (2*m)
                data_y = (y-d) % n
            data_array[data_x][data_y] ^= andval
    return (data_array,check_array)

def map_data_qubits_72(n):
    p = n // 6
    q = n % 6
    if 0 <= n < 36:
        x = 12 - 2 * (q + 1)
        y = (4 - p) % 6
    if 36 <= n < 72:
        x = (9 - 2 * (q + 1)) % 12
        y = (12 - p) % 6
    return [x, y]

def data_array_xor(m, n, array1, array2):
    xor_array = []
    for x in range(2*m):
        row = []
        for y in range(n):
            xor_value = array1[x][y] ^ array2[x][y]
            row.append(xor_value)
        xor_array.append(row)
    return xor_array

def whether_complex(actual_check_array, m, n):
    check_array=copy.deepcopy(actual_check_array)
    iscomplex = 0
    for x in range(m):
        for y in range(n):
            if(check_array[x][y]==1):
                iscomplex=1
                break
    return iscomplex

def main():
    m = 6
    n = 6
    a = 2
    b = 3
    c = 3
    d = 2
    w = 6
    p_error = 0.01
    num_trials = 1

    num_same = 0
    num_same_noncomplex = 0
    num_trials_nocomplex = 0

    parser = argparse.ArgumentParser()
    # parser.add_argument("code", default="72-12-6-w-6.pkl")
    parser.add_argument("-p", nargs="+", type=float, default=p_error)
    parser.add_argument("-i", "--idle", action="count", default=0)
    parser.add_argument("-t1", type=float, default=1e7)
    parser.add_argument("-t2", type=float, default=1e7)
    args = parser.parse_args()
    # n, k, d, w = parse.parse("codes/{}-{}-{}-w-{}.pkl", args.code)
    code = pkl.load(open("codes/72-12-6-w-6.pkl", "rb"))
    # print(f"Simulating code: [{code.n}, {code.k}, {code.d}] Weight {w}")
    # print(code.stim_circ(args.p, args.p, args.p, args.t1, args.t2, dec_type="Z", idle=args.idle, num_rounds=code.d))
    circuit = code.stim_circ(args.p, args.p, args.p, args.t1, args.t2, dec_type="Z", idle=args.idle, num_rounds=code.d)
    detector_coordinates = circuit.get_detector_coordinates()
    t_max = int(len(detector_coordinates)/(m*n))

    dem = circuit.detector_error_model()
    sampler = dem.compile_sampler()
    # detectors, observables, _ = sampler.sample(1)
    # stim_syndrome = np.flatnonzero(detectors)
    # stim_obs_synd_vector = 

    classes = generate_disjoint_check_classes(m, n, a, b, c, d)

    for trial in range(num_trials): 
        print("trial", trial+1)
        detectors, observables, _ = sampler.sample(1)
        stim_syndrome = np.flatnonzero(detectors)
        stim_obs_par_vector = []
        for i in range(len(observables[0])):
            stim_obs_par_vector.append(int(observables[0][i]))

        check_array_array = generate_check_array_array_72(m, n, t_max, stim_syndrome, detector_coordinates)
        decoded_check_array_array, correction_array_array = generate_decoded_array_array(check_array_array, classes, m, n, a, b, c, d)
        # for t in range(len(decoded_check_array_array)):
            # print("Round", t)
            # print("The original physical layout of X checks:")
            # for row in range(len(check_array_array[t])):
            #     print(check_array_array[t][row])
            # print("The physical layout of X checks after the Pinball decoder:")
            # for row in range(len(decoded_check_array_array[t])):
            #     print(decoded_check_array_array[t][row])
            # print("The physical layout of data qubit correction array:")
            # for row in range(len(correction_array_array[t])):
            #     print(correction_array_array[t][row])

        # print(len(decoded_check_array_array))
        # print(decoded_check_array_array[len(decoded_check_array_array)-1])

        iscomplex = whether_complex(decoded_check_array_array[len(decoded_check_array_array)-1], m, n)
        if(iscomplex!=1): 
            num_trials_nocomplex += 1

            XOR_correction_array = correction_array_array[0]
            for t in range(len(correction_array_array)-1):
                XOR_correction_array = data_array_xor(m, n, XOR_correction_array, correction_array_array[t+1])
            # print("The XOR physical layout of data qubit correction array:")
            # for row in range(len(XOR_correction_array)):
            #     print(XOR_correction_array[row])

            filename = f"circuit_{code.n}.txt"
            with open(filename, "w") as f:
                f.write(str(circuit))

            with open(filename, "r") as f:
                lines = f.readlines()
            m_line = ""
            for line in reversed(lines):
                if line.startswith("M"):
                    m_line = line
                    break
            m_nums = [int(token) for token in m_line.strip().split()[1:]]

            obs_ind_array = []
            obs_coor_array = []
            obs_synd_array = []
            obs_par_vector = []
            for i, line in enumerate(lines):
                if line.startswith("OBSERVABLE_INCLUDE"):
                    current_obs_ind = []
                    current_obs_coor = []
                    current_obs_synd = []
                    current_obs_par = 0
                    indices = re.findall(r"rec\[-(\d+)\]", lines[i])
                    for index_str in indices:
                        index = int(index_str)
                        value = m_nums[-index]
                        current_obs_ind.append(value)
                        coordinate = map_data_qubits_72(value)
                        current_obs_coor.append(coordinate)
                        syndrome = XOR_correction_array[coordinate[0]][coordinate[1]]
                        current_obs_synd.append(syndrome)
                        current_obs_par ^= syndrome 
                    obs_ind_array.append(current_obs_ind)
                    obs_coor_array.append(current_obs_coor)
                    obs_synd_array.append(current_obs_synd)
                    obs_par_vector.append(current_obs_par)

            # print("The observable coordinate array:")
            # for i in range(len(obs_coor_array)):
            #     print(obs_coor_array[i])
            
            # print("The observable syndrome array by Pinball:")
            # for i in range(len(obs_synd_array)):
            #     print(obs_synd_array[i])

            # stim_obs_synd_array = get directly from circuit
            # print("The observable syndrome array directly from circuit:")
            # for i in range(len(stim_obs_synd_array)):
            #     print(stim_obs_synd_array[i])

            # num_distinct = 0
            # for i in range(len(obs_synd_array[0])):
            #     for j in range(len(obs_synd_array[1])):
            #         if obs_synd_array[i][j] != stim_obs_synd_array[i][j]:
            #             num_distinct += 1

            # if num_distinct == 0:
            #     num_same += 1

            # print("stim observable parity:", stim_obs_par_vector)
            # print("pinball observable parity:", obs_par_vector)

            num_distinct = 0
            for i in range(len(obs_par_vector)):
                if obs_par_vector[i] != stim_obs_par_vector[i]:
                    num_distinct += 1

            if num_distinct == 0:
                num_same_noncomplex += 1
        
        else:
            num_trials_complex += 1 

    # print("probability of error:", p_error)
    # print("number of trials:", num_trials)
    # print("number of noncomplex:", num_trials_nocomplex)
    # print("number of same results within noncomplex trials:", num_same_noncomplex)
    # print("number of complex:", num_trials_complex)

    tasks = generate_task(code, args.p, args.idle, w, args.t1, args.t2)
    samples = sinter.collect(
        num_workers=48,
        max_shots=1_000_000,
        max_errors=50,
        tasks=tasks,
        count_observable_error_combos=True,
        decoders=["bposd"],
        custom_decoders={
            "bposd": BPOSD(
                max_iter=10_000, bp_method="ms", osd_order=10, osd_method="osd_cs"
            )
        },
    )
    pkl.dump(samples, open(f"C:/Users/robin/Desktop/gbstim-pytsp-master-original/gbstim-pytsp-master/scripts/results/{n}-{k}-{d}-w-{w}-samples.pkl", "wb"))


if __name__ == "__main__":
    print("heloo")
    main()
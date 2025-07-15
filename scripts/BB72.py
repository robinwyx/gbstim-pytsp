import random
import copy
import math
import numpy as np
import ldpc.codes
from ldpc import BpOsdDecoder
import scipy

def generate_data_array(m, n, error_probability):
    data_array = []
    for _ in range(2*m):
        row = [1 if random.random() < error_probability else 0 for _ in range(n)]
        data_array.append(row)
    return data_array

def generate_check_array(actual_data_array, m, n, a, b, c, d):
    data_array=copy.deepcopy(actual_data_array)
    check_array = []
    for x in range(m):
        row = []
        for y in range(n):
            parity=0 
            parity^=data_array[(2*x-1) % (2*m)][y % n] # top L
            parity^=data_array[(2*x+1) % (2*m)][y % n] # bottom L
            parity^=data_array[(2*x) % (2*m)][(y-1) % n] # left R
            parity^=data_array[(2*x) % (2*m)][y % n] # right R
            parity^=data_array[(2*(x-a+1)-1) % (2*m)][(y-b) % n] # far L
            parity^=data_array[(2*(x-c)) % (2*m)][(y-d) % n] # far R
            row.append(parity)
        check_array.append(row)
    return check_array

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

def whether_complex(actual_check_array, m, n):
    check_array=copy.deepcopy(actual_check_array)
    iscomplex = 0
    for x in range(m):
        for y in range(n):
            if(check_array[x][y]==1):
                iscomplex=1
                break
    return iscomplex

def array_xor(m, n, array1, array2):
    xor_array = []
    for x in range(2*m):
        row = []
        for y in range(n):
            xor_value = array1[x][y] ^ array2[x][y]
            row.append(xor_value)
        xor_array.append(row)
    return xor_array

def how_many_n_in_x(x,n):
    r = x%n
    return int((x-r)/n)

def generate_data_position(m, n, a, d):
    data_position = [[]*2]*(2*m*n)
    for i in range(m*n):
        count = how_many_n_in_x(i,m)
        data_position[i] = [(2*((i%m)-a+1)-1)%(2*m), count%(n)]
        data_position[i+(m*n)] = [(2*(i%m))%(2*m), (count-d)%(n)]
    return data_position

def generate_parity_check_matrix(m, n):
    parity_check_matrix = []
    for row in range(m*n):
        new_row = [0]*(2*m*n)
        for section in range(n):
            if (how_many_n_in_x(row,m)==section):
                if (how_many_n_in_x(row+1,m)==section):
                    new_row[row+1] = 1
                else:
                    new_row[row+1-m] = 1
                if (how_many_n_in_x(row+2,m)==section):
                    new_row[row+2] = 1
                else:
                    new_row[row+2-m] = 1
                if (how_many_n_in_x(row+3,m)==section):
                    new_row[row+3+m*n] = 1
                else:
                    new_row[row+3-m+m*n] = 1
        new_row[(row+18)%(m*n)] = 1
        new_row[(row+6)%(m*n)+m*n] = 1
        new_row[(row+12)%(m*n)+m*n] = 1
        parity_check_matrix.append(new_row)
    # for i in range(len(parity_check_matrix)):
    #     print(parity_check_matrix[i])
    return np.array(parity_check_matrix)

def generate_syndrome_vector(actual_check_array, m, n):
    check_array=copy.deepcopy(actual_check_array)
    syndrome_vector = [0]*(m*n)
    for i in range(m*n):
        count = how_many_n_in_x(i,m)
        syndrome_vector[i] = check_array[i%m][count%(n)]
    # data_array=copy.deepcopy(actual_data_array)
    # data_position = generate_data_position(m, n, a, b, c, d)
    # data_vector = [0]*(2*m*n)
    # for i in range(2*m*n):
    #     data_vector[i] = data_array[data_position[i][0]][data_position[i][1]]
    # syndrome_test = np.dot(parity_check_matrix, data_vector)%2
    # are_equal = syndrome_vector == syndrome_test
    return np.array(syndrome_vector)

def pure_bp(syndrome_vector, parity_check_matrix, error_probability):
    H = parity_check_matrix
    bp_osd = BpOsdDecoder(
                H,
                error_rate = error_probability,
                bp_method = 'product_sum',
                max_iter = 10,
                schedule = 'serial',
                osd_method = 'osd_cs', #set to OSD_0 for fast solve
                osd_order = 2
            )
    syndrome = syndrome_vector
    bp_decoded = bp_osd.decode(syndrome)
    # bp_decoded_syndrome = H@bp_decoded % 2
    # return syndrome, bp_decoded, bp_decoded_syndrome
    return bp_decoded

def wrapper(m, n, a, b, c, d, error_probability, num_trials):
    num_same = 0
    num_same_noncomplex = 0
    num_trials_nocomplex = 0
    classes = generate_disjoint_check_classes(m, n, a, b, c, d)
    for trial in range(num_trials):
        print("trial", trial)
        data_array = generate_data_array(m, n, error_probability)
        check_array = generate_check_array(data_array, m, n, a, b, c, d)

        data_position = generate_data_position(m, n, a, d)
        data_vector = [0]*(2*m*n)
        for i in range(2*m*n):
            data_vector[i] = data_array[data_position[i][0]][data_position[i][1]]

        H = generate_parity_check_matrix(m, n)

        # pure BP
        pureBP_syndrome_vector = generate_syndrome_vector(check_array, m, n)
        pureBP_decoded_data_vector = pure_bp(pureBP_syndrome_vector, H, error_probability)
        pureBP_xor_decoded_data_vector = []
        for i in range(len(pureBP_decoded_data_vector)):
            pureBP_xor_decoded_data_vector.append(pureBP_decoded_data_vector[i]^data_vector[i]) 

        # Clique + BP
        decoded_data_array,decoded_check_array = generate_decoded_data_array(check_array, classes, m, n, a, b, c, d)
        xor_decoded_data_array = array_xor(m, n, data_array, decoded_data_array)
        xor_decoded_data_vector = [0]*(2*m*n)
        for i in range(2*m*n):
            xor_decoded_data_vector[i] = xor_decoded_data_array[data_position[i][0]][data_position[i][1]]
        iscomplex = whether_complex(decoded_check_array, m, n)
        if(iscomplex!=1): 
            num_trials_nocomplex += 1 
            num_same += int(np.array_equal(pureBP_xor_decoded_data_vector, xor_decoded_data_vector))
            num_same_noncomplex += int(np.array_equal(pureBP_xor_decoded_data_vector, xor_decoded_data_vector))
        else:
            cxBP_syndrome_vector = generate_syndrome_vector(decoded_check_array, m, n)
            cxBP_decoded_data_vector = pure_bp(cxBP_syndrome_vector, H, error_probability)
            cxBP_xor_decoded_data_vector = []
            for i in range(len(cxBP_decoded_data_vector)):
                cxBP_xor_decoded_data_vector.append(cxBP_decoded_data_vector[i]^xor_decoded_data_vector[i]) 
            num_same += int(np.array_equal(pureBP_xor_decoded_data_vector, cxBP_xor_decoded_data_vector))

        # print("trial", trial, "data array:")
        # for i in range(len(data_array)):
        #     print(data_array[i])
        # print("trial", trial, "check array:")
        # for i in range(len(check_array)):
        #     print(check_array[i])
        # print("trial", trial, "decoded data array:")
        # for i in range(len(decoded_data_array)):
        #     print(decoded_data_array[i])
        # print("trial", trial, "decoded check array:")
        # for i in range(len(decoded_check_array)):
        #     print(decoded_check_array[i])  
        # print("trial", trial, "XOR data array:")
        # for i in range(len(xor_decoded_data_array)):
        #     print(xor_decoded_data_array[i])

    for cls in range(len(classes)):
        print(len(classes[cls]))
    print("probability of error:", error_probability)
    print("number of trials:", num_trials)
    print("number of same results:", num_same)
    print("number of noncomplex:", num_trials_nocomplex)
    print("number of same results within noncomplex trials:", num_same_noncomplex)



m=6
n=6
a=2
b=3
c=3
d=2
error_probability = 0.01
num_trials = 100000
wrapper(m, n, a, b, c, d, error_probability, num_trials)

# for error_probability in [0.01, 0.005, 0.001, 0.0005, 0.0001]
#         wrapper(m, n, a, b, c, d, error_probability, num_trials)

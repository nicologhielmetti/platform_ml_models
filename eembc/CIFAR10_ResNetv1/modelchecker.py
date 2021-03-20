from itertools import product
import csv
from resnet_v1_eembc import resnet_v1_eembc

fi = [2, 4, 8, 16, 32, 64]
f1_1 = [2, 4, 8, 16, 32, 64]
f1_2 = [2, 4, 8, 16, 32, 64]
f2_1 = [2, 4, 8, 16, 32, 64]
f2_2 = [2, 4, 8, 16, 32, 64]
f2_3 = [2, 4, 8, 16, 32, 64]
f3_1 = [2, 4, 8, 16, 32, 64]
f3_2 = [2, 4, 8, 16, 32, 64]
f3_3 = [2, 4, 8, 16, 32, 64]

ki = [1,2,3]
k1_1 = [1,2,3]
k1_2 = [1,2,3]
k2_1 = [1,2,3]
k2_2 = [1,2,3]
k2_3 = [1,2,3]
k3_1 = [1,2,3]
k3_2 = [1,2,3]
k3_3 = [1,2,3]

si = [1,2,3,4]
s1_1 = [1,2,3,4]
s1_2 = [1,2,3,4]
s2_1 = [1,2,3,4]
s2_2 = [1,2,3,4]
s2_3 = [1,2,3,4]
s3_1 = [1,2,3,4]
s3_2 = [1,2,3,4]
s3_3 = [1,2,3,4]

configs_comb = product(fi, f1_1, f1_2, f2_1, f2_2, f2_3, f3_1, f3_2, f3_3, ki, k1_1, k1_2, k2_1, k2_2, k2_3, k3_1, k3_2, k3_3, si, s1_1, s1_2, s2_1, s2_2, s2_3, s3_1, s3_2, s3_3)
configs_comb = list(configs_comb)
print(len(configs_comb))
valid_configs = []

for c in configs_comb:
    try:
        model = resnet_v1_eembc(fi=c[0], f1_1=c[1], f1_2=c[2], f2_1=c[3], f2_2=c[4], f2_3=c[5], f3_1=c[6], f3_2=c[7], f3_3=c[8],ki=c[9], k1_1=c[10], k1_2=c[11], k2_1=c[12], k2_2=c[13], k2_3=c[14], k3_1=c[15], k3_2=[16], k3_3=c[17], si=c[18], s1_1=c[19], s1_2=c[20], s2_1=c[21], s2_2=c[22], s2_3=c[23], s3_1=c[24], s3_2=[25],s3_3=c[26])
    except ValueError:
        #print(c, "not valid")
        continue
    print(c, "valid")
    valid_configs.append(c)

print("valid configs")
print(valid_configs)

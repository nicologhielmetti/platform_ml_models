from itertools import product

filters_InL = [4,8,16,32,64]
kernelsize_InL = [1,2,3,4]
strides_InL = [1,2,3,4]

configs_comb = product(filters_InL,kernelsize_InL,strides_InL)
configs_comb = list(configs_comb)

print(configs_comb)
print("\n",len(configs_comb))
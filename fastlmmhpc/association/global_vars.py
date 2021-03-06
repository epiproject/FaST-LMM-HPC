##### GPU ######
global cpu_dot
global gpu_free
global cpy_fortran
global gpu_transfer
global gpu_dot

global n_cpu_mult
global n_gpu_mult1
global n_gpu_mult2
global n_gpu_mult3

n_cpu_mult    = 0
n_gpu_mult1   = 0
n_gpu_mult2   = 0
n_gpu_mult3   = 0

cpu_dot      = 0
cpy_fortran  = 0
gpu_transfer = 0
gpu_dot      = 0
gpu_free     = 0

##### GPU ######

global first_section
global second_section

global call_func
global other_func

global snd_col1
global snd_log1
global snd_col2
global snd_log2
global snd_chi
global snd_res

global log_time
global log_time2

global nLL_p1
global nLL_p2
global nLL_p3


global gpu_mult
global cpu_mult

global transfer_t
global mult_t

global maxM

maxM = False

first_section  = 0
second_section = 0

call_func  = 0
other_func = 0

snd_col1 = 0
snd_log1 = 0
snd_col2 = 0
snd_log2 = 0
snd_chi  = 0
snd_res  = 0

log_time  = 0
log_time2 = 0
nl_time   = 0

nLL_p1 = 0
nLL_p2 = 0
nLL_p3 = 0


gpu_mult = 0
cpu_mult = 0

transfer_t = 0
mult_t = 0

'''
global facum_2
global facum_3
global facum_4

global out_loop
global in_loop

global run_mode

global NEW_MODE
global NORMAL_MODE

global facum_1


facum_1 = 0

facum_2 = 0
facum_3 = 0
facum_4 = 0

out_loop = 0
in_loop  = 0

NEW_MODE    = 1
NORMAL_MODE = 0

run_mode = NORMAL_MODE
'''

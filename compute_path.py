"""
This file will be called by a .sh file to compute paths of the following:
1. An alpha particle and an H^- ion moving in a uniform magnetic field
2. An alpha particle moving in the field of a magnetic dipole
3. An alpha particle moving in a 'magnetic bottle' field.
"""



import numpy as np
import AdvLabComputation as alc
import time


start_time = time.time()

field_type = input("Enter the type of field configuration desired (uni_field/dipole_field/dipole_field_23/bottle_field/const_mag_const_elec): ")

while field_type != 'uni_field' and field_type != 'dipole_field' and field_type != 'dipole_field_23' and field_type != 'bottle_field' and field_type!= 'const_mag_const_elec':
    field_type = input("Configuration type not valid. Please enter uni_field, dipole_field, or bottle_field): ")


alpha_charge = 2 * 1.602 * 10**(-19)
alpha_mass = 6.446573357 * 10**(-27)
hyd_charge = -1.602 * 10**(-19)
hyd_mass = 1.67262158 * 10**(-27)

alpha_sys = alc.System(mass_particle = alpha_mass, charge_particle = alpha_charge)
hyd_sys = alc.System(mass_particle = hyd_mass, charge_particle = hyd_charge)



if field_type == 'uni_field':
    timelength = 3*10**(-2)
    num_of_iters = 10**5
    
    alpha_path = alpha_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,0,0]), init_vel = np.array([31000, 0, 0]), config = 'const_mag')

    hyd_path = hyd_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,0,0]), init_vel = np.array([31000, 0, 0]), config = 'const_mag')



elif field_type == 'dipole_field':
    timelength = 1
    num_of_iters = 10**5
    
    alpha_path = alpha_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,-8,2]), init_vel = np.array([0, 100, 0]), config = 'mag_dipole')

    hyd_path = hyd_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,-8,2]), init_vel = np.array([0, 100, 0]), config = 'mag_dipole')

    field_lines = alpha_sys.make_field_lines(config = 'mag_dipole', box_length = 20, box_cuts = 15)
    

elif field_type == 'dipole_field_23':
    timelength = 1
    num_of_iters = 10**5
    
    alpha_path = alpha_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,-8, 0]), init_vel = np.array([0, 100, 0]), config = 'mag_dipole_23')

    hyd_path = hyd_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,-8,0]), init_vel = np.array([0, 100, 0]), config = 'mag_dipole_23')

    field_lines = alpha_sys.make_field_lines(config = 'mag_dipole_23', box_length = 20, box_cuts = 15)
    

elif field_type == 'bottle_field':
    timelength = 1
    num_of_iters = 10**5
    
    alpha_path = alpha_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([-5,0,0]), init_vel = np.array([0, 0, 100]), config = 'mag_bottle')

    hyd_path = hyd_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([-5,0,0]), init_vel = np.array([0, 0, 100]), config = 'mag_bottle')

    field_lines = alpha_sys.make_field_lines(config = 'mag_bottle', box_length = 40, box_cuts = 15)


elif field_type == 'const_mag_const_elec':
    timelength = 1
    num_of_iters = 10**5

    alpha_path = alpha_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([0,0,0]), init_vel = np.array([0, 0, 1]), config = 'const_mag_const_elec')

    hyd_path = hyd_sys.solve_path(timelength = timelength, num_of_iters = num_of_iters, init_pos = np.array([-5,0,0]), init_vel = np.array([0, 0, 1]), config = 'const_mag_const_elec')
    

np.savez('alpha_path_{}.npz'.format(field_type), alpha_path)
np.savez('hyd_path_{}.npz'.format(field_type), hyd_path)

if field_type != 'uni_field' and field_type != 'const_mag_const_elec':
    np.savez('field_lines_{}.npz'.format(field_type), field_lines)


print("--- {} seconds---".format(time.time() - start_time))

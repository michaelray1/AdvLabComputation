"""
Michael Ray
File containing code for the advanced lab computational section. 
This file will contain all the classes and functions needed
for the questions. They'll then be implemented in a jupyter
notebook.
"""

import numpy as np


class System:
    """
    system object. This will be how we track the trajectory
    of the particle and code in all the relevant forces and such.
    """

    def __init__(self, mass_particle, charge_particle):
        """
        init function will set up the charge of the particle, the
        magnetic field, and electric field as attributes of system.
        
        Parameters:
        mass_particle: Give a real number that is the mass of the
        particle of interest
        
        charge_particle: Give a real number which is the charge of
        whatever particle we are interested in.
        """
        self.mass_particle = mass_particle
        self.charge_particle = charge_particle


        

    def do_step(self, start_pos, start_vel, timestep, elec_field, mag_field):
        """
        This function runs one step in the process of calculating the
        motion of the particle. The step first calculates the force on 
        the particle, then solves a simple kinematics equation to learn
        where the particle will be one unit of time from now.

        Parameters:
        start_pos: Give a numpy array of dimension 3 by 1 which represents
        the initial position of the particle in cartesian coordinates

        start_vel: Give a numpy array of dimension 3 by 1 which represents
        the initial velocity of the particle.
        
        timestep: Give a real number which is the size of the time step that
        the function will use to determine the next position.

        elec_field: Give a 3 by 1 numpy array which is the electric field
        for this iteration

        mag_field: Give a 3 by 1 numpy array which is the magnetic field
        for this iteration.

        Returns:
        fin_position: Where the particle is after the timestep
        fin_velocity: Velocity of the particle after the timestep
        """
        force = self.charge_particle * elec_field + self.charge_particle * np.cross(start_vel, mag_field)
        accel = force/(self.mass_particle)
        fin_position = start_pos + start_vel * timestep + (1/2) * accel * timestep**2
        fin_velocity = start_vel + accel * timestep

        return fin_position, fin_velocity



    
    def solve_path(self, timelength, num_of_iters, init_pos, init_vel, config):
        """
        Ths function will map out the motion of the particle
        by iterating the do_step function.

        Parameters:
        timelength: Give a real number which is the total time you want
        to simulate
        
        num_of_iters: Give the number of iterations you want to do in order
        to solve for the motion of the particle
        
        init_pos: Give a numpy array of dimension 3 by 1. This represents
        the position the particle starts in.
        
        init_vel: Give a numpy array of dimension 3 by 1. This represents
        the velocity the particle starts with.
        
        config: Give a string which is equal to one of the options
        below. This indicates what magnetic/electric field configuration
        we are working with. Options for configuration are:
        1. const_mag
        2. mag_dipole
        3. mag_dipole_23
        4. mag_bottle
        5. const_mag_const_elec
        These 5 options implement the 5 different configurations
        needed to answer all questions in the lab.

        Returns:
        pos_data: a list of all the positions that the particle was calculated
        to pass through
        
        vel_data: a list of all the velocities that the particle was calculated
        to have during the simulation.
        """
        timestep = timelength / num_of_iters
        pos = init_pos
        vel = init_vel

        pos_data = []
        vel_data = []
        pos_data.append(pos)
        vel_data.append(vel)

        i = 0
        while i < num_of_iters:
            if config == 'const_mag':
                e_field = np.array([0,0,0])
                b_field = np.array([0,0,10**(-4)])

            elif config == 'mag_dipole':
                mu = np.array([0, 0, 10**4])
                e_field = np.array([0,0,0])
                b_field = self.mag_dipole_field(r = pos, position = np.array([0,0,0]), mu = mu)

            elif config == 'mag_dipole_23':
                mu = np.array([0, 10**4 * np.sin(23 * (np.pi/180)), 10**4 * np.cos(23 * (np.pi/180))])
                e_field = np.array([0,0,0])
                b_field = self.mag_dipole_field(r = pos, position = np.array([0,0,0]), mu = mu)
                
            elif config == 'mag_bottle':
                mu = np.array([0, 0, 10**4])
                e_field = np.array([0,0,0])
                first_b = self.mag_dipole_field(r = pos, position = np.array([0, 0, 10]), mu = mu)
                second_b = self.mag_dipole_field(r = pos, position = np.array([0, 0, -10]), mu = mu)
                b_field = first_b + second_b

            elif config == 'const_mag_const_elec':
                e_field = np.array([-0.01, 0, 0])
                b_field = np.array([0,0,10**(-4)])

            else:
                raise ValueError("Configuration is not valid. Please input one of the following for config: const_mag, mag_dipole, mag_bottle, or const_mag_const_elec.")
            
            next_pos, next_vel = self.do_step(start_pos = pos, start_vel = vel, timestep = timestep, elec_field = e_field, mag_field = b_field)
            pos_data.append(next_pos)
            vel_data.append(next_vel)
            pos = next_pos
            vel = next_vel
            i += 1

        return pos_data




    def mag_dipole_field(self, r, position = np.array([0,0,0]), mu = np.array([0,0,0])):
        """
        This function numerically solves for the magnetic field of a dipole
        mu at a position r.

        Parameters:
        r - Give a 3 by 1 numpy array which represents the point at which
        you want to evaluate the dipole field. Give this in cartesian
        ccords

        position - Give a 3 by 1 numpy array which represents the point at
        which the dipole is located. This is in cartesian coords. Default
        value is the origin.

        mu - Give a 3 by 1 numpy array which is the dipole vector in
        cartesian coords. Default is no dipole at all.

        

        Returns:
        The B field corresponding to the point r due to the dipole mu.
        """

        mag_mu = np.linalg.norm(mu)
        mag_r = np.linalg.norm(r - position)
        b_field = (mag_mu / (4 * np.pi)) * ((3 * (r - position) * np.dot(mu, r - position) / (mag_r**5)) - mu / (mag_r**3))

        return b_field



    def make_field_lines(self, config, box_length):
        """
        This function will take in a field configuration and produce
        a numpy array which has size 3 by 2 by 1000 where 3 represents the
        3 components of the position or field  in cartesian coordinates, 
        2 represents the fact that we need the position and vector field,
        and 1000 is the number of positions at which we will calculate
        the field.

        Parameters:
        config- Give a string which is equal to one of the options                                                              
        below. This indicates what magnetic/electric field configuration                                                        
        we are working with. Options for configuration are:                                                                     
        1. const_mag                                                                                                            
        2. mag_dipole                                                                                                           
        3. mag_dipole_23                                                                                                        
        4. mag_bottle                                                                                                           
        5. const_mag_const_elec                                                                                                 
        These 5 options implement the 5 different configurations                                                                
        needed to answer all questions in the lab.
        
        box_length- give a real number which is the size of the box
        you want to simulate. 1000 field lines will be produced which
        represent the box length cut into 10 segments along each of
        the 3 cartesian directions.

        Returns
        A numpy array of shape 3 by 2 by 1000. The 3 represents the
        three cartesian coordinates. The 2 represents the fact
        that we need position data as well as field data. The
        1000 is the number of points at which we calculate the 
        field lines.
        """

        step = box_length/10

        #Set up the positions at which we will calculate the field
        field = np.empty([3, 2, 1000])
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    field[:, 0, 100*i + 10*j + k] = [-box_length/2 + i*step, -box_length/2 + j*step, -box_length/2 + k*step]
        
        if config == 'const_mag':
            for i in range(len(field[0, 0, :])):
                field[:, 1, i] = np.array([0,0,10**(-4)])

        elif config == 'mag_dipole':
            mu = np.array([0, 0, 10**4])
            for i in range(len(field[0, 0, :])):
                field[:, 1, i] = self.mag_dipole_field(r = field[:, 0, i], position = np.array([0,0,0]), mu = mu)

        elif config == 'mag_dipole_23':
            mu = np.array([0, 10**4 * np.sin(23 * (np.pi/180)), 10**4 * np.cos(23 * (np.pi/180))])
            for i in range(len(field[0, 0, :])):
                field[:, 1, i] = self.mag_dipole_field(r = field[:, 0, i], position = np.array([0,0,0]), mu = mu)

        elif config == 'mag_bottle':
            mu = np.array([0, 0, 10**4])
            for i in range(len(field[0, 0, :])):
                first_field = self.mag_dipole_field(r = field[:, 0, i], position = np.array([0, 0, -10]), mu = mu)
                second_field = self.mag_dipole_field(r = field[:, 0, i], position = np.array([0, 0, 10]), mu = mu)
                field[:, 1, i] = first_field + second_field

        else:
            raise ValueError("Configuration is not valid. Please input one of the following for config: const_mag, mag_dipol\
e, mag_bottle, or const_mag_const_elec.")



        return field

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

    def __init__(self, mass_particle, charge_particle, mag_field, elec_field):
        """
        init function will set up the charge of the particle, the
        magnetic field, and electric field as attributes of system.
        
        Parameters:
        mass_particle: Give a real number that is the mass of the
        particle of interest
        
        charge_particle: Give a real number which is the charge of
        whatever particle we are interested in.

        mag_field: Give a numpy array of dimension 3 by 1. 
        This is the magnetic field given in cartesian coordinates

        elec_field: Same thing as mag_field but for the electric field.
        """
        self.mass_particle = mass_particle
        self.charge_particle = charge_particle
        self.mag_field = mag_field
        self.elec_field = elec_field


    def do_step(self, start_pos, start_vel, timestep):
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

        Returns:
        fin_position: Where the particle is after the timestep
        fin_velocity: Velocity of the particle after the timestep
        """
        force = self.charge_particle * self.elec_field + self.charge_particle * np.cross(start_vel, self.mag_field)
        accel = force / self.mass_particle
        fin_position = start_pos + start_vel * timestep + (1/2) * accel * timestep**2
        fin_velocity = start_vel + accel * timestep

        return fin_position, fin_velocity


    def solve_path(self, timelength, num_of_iters, init_pos, init_vel):
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
        i = 0
        while i < num_of_iters:
            next_pos, next_vel = self.do_step(start_pos = pos, start_vel = vel, timestep = timestep)
            pos_data.append(next_pos)
            vel_data.append(next_vel)
            pos = next_pos
            vel = next_vel
            i += 1

        return pos_data, vel_data

import pandas as pd
import numpy as np
from rcplant import *
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

def check_PP(maxima):
    p1, p2, p3 = False, False, False
    for index, value in maxima.items():
        if (index > 1340 and index < 1400):
            if (value < 0.2 and value > 0.05): p1 = True
        if (index > 1420 and index < 1480):
            if (value < 0.2 and value > 0.05): p2 = True
        if (index > 2800 and index < 3000):
            if (value < 0.25 and value > 0.1): p3 = True
    return p1 and p2 and p3
    

def check_PS(maxima):
    p1, p2, p3, p4, p5 = 0, 0, 0, 0, 0
    for index, value in maxima.items():
        if(index > 1430 and index < 1470): p1 += 1 
        if(index > 1470 and index < 1510): p2 += 1 
        if(index > 1580 and index < 1620): p3 += 1
        if(index > 2900 and index < 2940): p4 += 1
        if(index > 3000 and index < 3050): p5 += 1 
        if(p1 + p2 + p3 + p4 + p5 >= 4):
            return True

def check_PC(maxima):
    p1, p2, p3 = False, False, False
    for index, value in maxima.items():
        if (value > 0.1 and value < 0.5):
            if (index > 1250 and index < 1260): p1 = True
            if (index > 1500 and index < 1510): p2 = True
            if (index > 1750 and index < 1790): p3 = True

    return p1 and p2 and p3

def check_HDPE(maxima):
    p1, p2 = False, False
    for index, value in maxima.items():
        if (index > 1450 and index < 1480):
            p1 = True
        if (index > 2850 and index < 2950):
            if (value > 0.45): p2 = True

    return p1 and p2

def check_LDPE(maxima):
    for index, value in maxima.items():
        if (index > 2850 and index < 2950):
            if (value < 0.45 and value > 0.25):
                return True

    return False

def check_PET(maxima):
    for index, value in maxima.items():
        if (index > 2000): 
            if (value > 0.1): return False
        if (index > 1690 and index < 1740):
            if (value > 0.2): return True
    return False

def check_Polyester(maxima):
    p1, p2 = False, False
    for index, value in maxima.items():
        if (index > 1690 and index < 1740):
            if (value > 0.2): 
                p1 = True
        if (index > 2700):
            p2 = True
    return p1 and p2


def user_sorting_function(sensors_output):
    # TODO: change this
    decision = { 1: Plastic.PVC }

    spectrum = sensors_output[1]['spectrum']

    # check for zero and shortcircuit
    if (spectrum.iloc[0] < 0.001 or spectrum.values.mean() < 0.01):
        return { 1: Plastic.Blank }

    z = np.polyfit(spectrum.keys().values[:-30], spectrum.values[:-30], 5)
    p = np.poly1d(z)

    # Get local maxima relative to 10 other points on each side
    iloc_max_wavenumbers = argrelextrema(spectrum.values, comparator=np.greater, order=10)[0]
    max_wavenumbers = spectrum.iloc[iloc_max_wavenumbers]

    # Add the local maxima of the first 10 points because it is missed in argrelextrema
    beginning = pd.Series([spectrum.iloc[-10:].max()], index=[spectrum.iloc[-10:].idxmax()])
    max_wavenumbers = pd.concat([max_wavenumbers, beginning]).drop_duplicates()

    # Remove every point that is 
    #   after 3250
    #   between 2000 and 2700
    #   less than 120% of the mean
    #   less than 2.7 times of the trendline
    threshold = max_wavenumbers.values.mean() * 120 / 100
    for index, value in max_wavenumbers.items():
        if (index > 3250 or
            (index > 2000 and index < 2700) or
            value < threshold or
            value < p(index) * 2.7):
            max_wavenumbers.drop(index, inplace=True)

    # Plot all the information on the spectrum (this is just visualisation)
    spectrum.plot()
    max_wavenumbers.plot(title=spectrum.name, style="v", color="red")
    plt.plot(spectrum.keys().values[:-30], p(spectrum.keys().values[:-30]) * 2.7, color="green")
    plt.hlines(y=spectrum.values.mean(), xmin=spectrum.keys().values[0], xmax=spectrum.keys().values[-1], linestyles='-', color='black')
    plt.hlines(y=threshold, xmin=spectrum.keys().values[0], xmax=spectrum.keys().values[-1], linestyles='-', color='blue')
    plt.show()
    
    for index, value in max_wavenumbers.items():
        print(f"{index} - {value}")

    if check_PS(max_wavenumbers):
        decision[1] = Plastic.PS
    elif check_PET(max_wavenumbers):
        decision[1] = Plastic.PET
    elif check_Polyester(max_wavenumbers):
        decision[1] = Plastic.Polyester
    elif check_HDPE(max_wavenumbers):
        decision[1] = Plastic.HDPE
    elif check_LDPE(max_wavenumbers):
        decision[1] = Plastic.LDPE
    elif check_PP(max_wavenumbers):
        decision[1] = Plastic.PP
    elif check_PC(max_wavenumbers):
        decision[1] = Plastic.PC

    print()
    return decision

def main():

    # simulation parameters
    conveyor_length = 1000  # cm
    conveyor_width = 100  # cm
    conveyor_speed = 5  # cm per second
    num_containers = 100
    sensing_zone_location_1 = 500  # cm
    sensors_sampling_frequency = 10  # Hz
    simulation_mode = 'testing'

    sensors = [
        Sensor.create(SpectrumType.FTIR, sensing_zone_location_1),
    ]

    conveyor = Conveyor.create(conveyor_speed, conveyor_length, conveyor_width)

    simulator = RPSimulation(
        sorting_function=user_sorting_function,
        num_containers=num_containers,
        sensors=sensors,
        sampling_frequency=sensors_sampling_frequency,
        conveyor=conveyor,
        mode=simulation_mode
    )

    elapsed_time = simulator.run()

    print(f'\nResults for running the simulation in "{simulation_mode}" mode:')

    for item_id, result in simulator.identification_result.items():
        print(result)

    print(f'Total missed containers = {simulator.total_missed}')
    print(f'Total sorted containers = {simulator.total_classified}')
    print(f'Total mistyped containers = {simulator.total_mistyped}')

    print(f'\n{num_containers} containers are processed in {elapsed_time:.2f} seconds')


if __name__ == '__main__':
    main()

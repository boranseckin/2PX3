import numpy as np
from rcplant import *
from pandas import Series
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

def check_HDPE(maxima):
    for index, value in maxima.items():
        if (index > 2800 and index < 2950):
            if (value > 0.45):
                print(f"{index} {value}")
                return True

    return False

def check_LDPE(maxima):
    for index, value in maxima.items():
        if (index > 2800 and index < 2950):
            if (value < 0.45 and value > 0.25):
                print(f"{index} {value}")
                return True

    return False

def check_general(maxima):
    temp = maxima.keys().values
    if not temp.any():
        return Plastic.PS
    for i in range(len(temp)):
        temp[i]=temp[i]//100
    if (17 in temp and (13 in temp or 12 in temp)):
        return Plastic.PU
    if (17 in temp and 14 in temp):
        return Plastic.PET
    if (17 in temp and 15 in temp):
        return Plastic.PC
    if (29 in temp):
        return Plastic.PP
    if (17 in temp):
        return Plastic.Polyester
    if (12 in temp):
        return Plastic.PVC
    if (14 in temp):
        return Plastic.PS

    return Plastic.Blank

def user_sorting_function(sensors_output):
    # TODO: change this
    decision = { 1: Plastic.Blank }

    spectrum = sensors_output[1]['spectrum']

    # check for zero and shortcircuit
    # if (spectrum.iloc[0] == 0):
    #     return { 1: Plastic.Blank }

    # Get local maxima relative to 10 other points on each side
    iloc_max_wavenumbers = argrelextrema(spectrum.values, comparator=np.greater, order=10)[0]
    max_wavenumbers = spectrum.iloc[iloc_max_wavenumbers]

    # Remove every point that is less than 75% of the mean
    threshold = 0.09
    for index, value in max_wavenumbers.items():
        if (value < threshold):
            max_wavenumbers.drop(index=index, inplace=True)

    # Plot the found points on the spectrum (this is just visualisation)
    if spectrum.name != "background":
        spectrum.plot()
        max_wavenumbers.plot(title=spectrum.name, style="v", color="red")
        print(max_wavenumbers.keys().values)
        plt.show()

    # TODO: add decision logic
    if check_HDPE(max_wavenumbers):
        decision[1] = Plastic.HDPE
    elif check_LDPE(max_wavenumbers):
        decision[1] = Plastic.LDPE
    else:
        decision[1] = check_general(max_wavenumbers)


    return decision


def main():

    # simulation parameters
    conveyor_length = 1000  # cm
    conveyor_width = 100  # cm
    conveyor_speed = 5  # cm per second
    num_containers = 100
    sensing_zone_location_1 = 500  # cm
    sensors_sampling_frequency = 1  # Hz
    simulation_mode = 'training'

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

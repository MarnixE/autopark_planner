import cv2
import numpy as np
# from time import sleep
import argparse

from environment import Environment, Parking1
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger


########################## How to run the code ################################################
# The different path planners can be selected by changing BiAstar to the file name of the relevant planner
# Furthermore, please uncomment the input arguments corresponding to RRT or A*
from BiAstar import PathPlanning, ParkPathPlanning, interpolate_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=0, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')

    # Arguments for A* / Bi-directional A*
    parser.add_argument('--parking', type=int, default=144, help='park position in parking1 out of 144')
    car_removed = np.random.randint(1,15,size=1)
    parser.add_argument('--parking_spots', type=int, default=car_removed, help='number of available parking spots')

    # Arguments for RRT / RRT*
    # parser.add_argument('--parking', type=int, default=2, help='park position in parking1 out of 2')
    # parser.add_argument('--parking_spots', type=int, default=1, help='number of available parking spots')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end = np.array([args.x_end, args.y_end])
    #############################################################################################

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking,args.parking_spots)
    end, obs = parking1.generate_obstacles()
    print('Generating Obstacles')

    ########################### initialization ##################################################
    print('Initializing controller ...')
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    print('planning park scenario ...')
    new_ends = []
    park_paths = []
    ensure_paths1 = []
    ensure_paths2 = []
    for end_pose in end:
        left_lane = park_path_planner.lane(int(start[0]), int(start[1]), int(end_pose[0]), int(end_pose[1]))
        if left_lane:
            new_end = np.array(end_pose) + np.array([5,0])
        else:
            new_end = np.array(end_pose) - np.array([5,0])
        new_ends.append(new_end)

    print('routing to destination ...')
    best_path = np.zeros(1000)
    idx = 0
    for new_end in new_ends:
        path = path_planner.plan_path(int(start[0]), int(start[1]), int(new_end[0]), int(new_end[1]))

        if len(best_path) > len(path):
            best_path = path
            best_idx = idx
        idx += 1

    new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),
                                                                                              int(start[1]),
                                                                                              int(end[best_idx][0]), int(end[best_idx][1]), best_path)

    path = np.vstack([best_path, ensure_path1])


    print('interpolating ...')
    interpolated_path = interpolate_path(best_path, sample_rate=5)
    interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

    env.draw_path(interpolated_path)
    env.draw_path(interpolated_park_path)

    final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    #############################################################################################

    ################################## control ##################################################
    print('driving to destination ...')
    for i, point in enumerate(final_path):

        acc, delta = controller.optimize(my_car, final_path[i:i + MPC_HORIZON])
        my_car.update_state(my_car.move(acc, delta))
        res = env.render(my_car.x, my_car.y, my_car.psi, delta)
        logger.log(point, my_car, acc, delta)
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('res.png', res * 255)

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    cv2.destroyAllWindows()

import numpy as np
import math
import scipy.interpolate as scipy_interpolate
from utils import angle_of_line


############################################## Functions ######################################################

def interpolate_b_spline_path(x, y, n_path_points, degree=3):
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)
    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)

def interpolate_path(path, sample_rate):
    choices = np.arange(0,len(path),sample_rate)
    if len(path)-1 not in choices:
            choices =  np.append(choices , len(path)-1)
    way_point_x = path[choices,0]
    way_point_y = path[choices,1]
    n_course_point = len(path)*3
    rix, riy = interpolate_b_spline_path(way_point_x, way_point_y, n_course_point)
    new_path = np.vstack([rix,riy]).T
    return new_path

################################################ Path Planner ################################################

class BidirectAStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        print('goal_node: ',goal_node)

        open_set_A, closed_set_A = dict(), dict()
        open_set_B, closed_set_B = dict(), dict()
        open_set_A[self.calc_grid_index(start_node)] = start_node
        open_set_B[self.calc_grid_index(goal_node)] = goal_node

        current_A = start_node
        current_B = goal_node
        meet_point_A, meet_point_B = None, None

        while 1:
            if len(open_set_A) == 0:
                print("Open set A is empty..")
                break

            if len(open_set_B) == 0:
                print("Open set B is empty..")
                break

            c_id_A = min(
                open_set_A,
                key=lambda o: self.find_total_cost(open_set_A, o, current_B))

            current_A = open_set_A[c_id_A]

            c_id_B = min(
                open_set_B,
                key=lambda o: self.find_total_cost(open_set_B, o, current_A))

            current_B = open_set_B[c_id_B]

            if current_A.x == current_B.x and current_A.y == current_B.y:
                print("Found goal")
                meet_point_A = current_A
                meet_point_B = current_B
                break

            # Remove the item from the open set
            del open_set_A[c_id_A]
            del open_set_B[c_id_B]

            # Add it to the closed set
            closed_set_A[c_id_A] = current_A
            closed_set_B[c_id_B] = current_B

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):

                c_nodes = [self.Node(current_A.x + self.motion[i][0],
                                     current_A.y + self.motion[i][1],
                                     current_A.cost + self.motion[i][2],
                                     c_id_A),
                           self.Node(current_B.x + self.motion[i][0],
                                     current_B.y + self.motion[i][1],
                                     current_B.cost + self.motion[i][2],
                                     c_id_B)]

                n_ids = [self.calc_grid_index(c_nodes[0]),
                         self.calc_grid_index(c_nodes[1])]

                # If the node is not safe, do nothing
                continue_ = self.check_nodes_and_sets(c_nodes, closed_set_A,
                                                      closed_set_B, n_ids)

                if not continue_[0]:
                    if n_ids[0] not in open_set_A:
                        # discovered a new node
                        open_set_A[n_ids[0]] = c_nodes[0]
                    else:
                        if open_set_A[n_ids[0]].cost > c_nodes[0].cost:
                            # This path is the best until now. record it
                            open_set_A[n_ids[0]] = c_nodes[0]

                if not continue_[1]:
                    if n_ids[1] not in open_set_B:
                        # discovered a new node
                        open_set_B[n_ids[1]] = c_nodes[1]
                    else:
                        if open_set_B[n_ids[1]].cost > c_nodes[1].cost:
                            # This path is the best until now. record it
                            open_set_B[n_ids[1]] = c_nodes[1]
        rx, ry = self.calc_final_bidirectional_path(
            meet_point_A, meet_point_B, closed_set_A, closed_set_B)

        return rx, ry

    # takes two sets and two meeting nodes and return the optimal path
    def calc_final_bidirectional_path(self, n1, n2, setA, setB):
        rx_A, ry_A = self.calc_final_path(n1, setA)
        rx_B, ry_B = self.calc_final_path(n2, setB)

        rx_A.reverse()
        ry_A.reverse()

        rx = rx_A + rx_B
        ry = ry_A + ry_B

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index

        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def check_nodes_and_sets(self, c_nodes, closedSet_A, closedSet_B, n_ids):
        continue_ = [False, False]
        if not self.verify_node(c_nodes[0]) or n_ids[0] in closedSet_A:
            continue_[0] = True

        if not self.verify_node(c_nodes[1]) or n_ids[1] in closedSet_B:
            continue_[1] = True

        return continue_

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def find_total_cost(self, open_set, lambda_, n1):
        g_cost = open_set[lambda_].cost
        h_cost = self.calc_heuristic(n1, open_set[lambda_])
        f_cost = g_cost + h_cost
        return f_cost

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = np.zeros((self.x_width,self.y_width),dtype=bool)

        for ix in range(self.x_width):
            x = ix*self.resolution + self.min_x
            for iy in range(self.y_width):
                y = iy*self.resolution + self.min_y
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d < self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


class PathPlanning:
    def __init__(self,obstacles):
        self.margin = 5
        # scale obstacles from env margin to pathplanning margin
        obstacles = obstacles + np.array([self.margin,self.margin])
        obstacles = obstacles[(obstacles[:,0]>=0) & (obstacles[:,1]>=0)]

        self.obs = np.concatenate([np.array([[0,i] for i in range(100+self.margin)]),
                                  np.array([[100+2*self.margin,i] for i in range(100+2*self.margin)]),
                                  np.array([[i,0] for i in range(100+self.margin)]),
                                  np.array([[i,100+2*self.margin] for i in range(100+2*self.margin)]),
                                  obstacles])

        self.ox = [int(item) for item in self.obs[:,0]]
        self.oy = [int(item) for item in self.obs[:,1]]
        self.grid_size = 1
        self.robot_radius = 4
        self.bi_a_star = BidirectAStarPlanner(self.ox, self.oy, self.grid_size, self.robot_radius)

    def plan_path(self,sx, sy, gx, gy):    
        rx, ry = self.bi_a_star.planning(sx+self.margin, sy+self.margin, gx+self.margin, gy+self.margin)
        rx = np.array(rx)-self.margin
        ry = np.array(ry)-self.margin
        path = np.vstack([rx,ry]).T
        return path

############################################### Park Path Planner #################################################

class ParkPathPlanning:
    def __init__(self, obstacles):
        self.margin = 0
        # sacale obstacles from env margin to pathplanning margin
        obstacles = obstacles + np.array([self.margin, self.margin])
        obstacles = obstacles[(obstacles[:, 0] >= 0) & (obstacles[:, 1] >= 0)]

        self.obs = np.concatenate([np.array([[0, i] for i in range(100 + self.margin)]),
                                   np.array([[100 + 2 * self.margin, i] for i in range(100 + 2 * self.margin)]),
                                   np.array([[i, 0] for i in range(100 + self.margin)]),
                                   np.array([[i, 100 + 2 * self.margin] for i in range(100 + 2 * self.margin)]),
                                   obstacles])

        self.ox = [int(item) for item in self.obs[:, 0]]
        self.oy = [int(item) for item in self.obs[:, 1]]
        self.grid_size = 5
        self.robot_radius = 4
        self.bi_astar = BidirectAStarPlanner(self.ox, self.oy, self.grid_size, self.robot_radius)

    def lane(self, sx, sy, gx, gy):
        Lane = (gx - 10) % 20 < 10
        return Lane

    def generate_park_scenario(self, sx, sy, gx, gy, path):
        # Compute the direction the car came from
        computed_angle = angle_of_line(path[-10][0],path[-10][1],path[-1][0],path[-1][1])
        left_lane = self.lane(sx, sy, gx, gy)
        self.s = 0
        self.l = 0
        self.d = 1
        self.w = 4

        if -math.atan2(0,-1) < computed_angle <= math.atan2(0,1) and left_lane:
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 + self.d + self.w
            y_ensure1 = y_ensure2 - self.l - self.s
            ensure_path1 = np.vstack(
                [np.repeat(x_ensure1, 2 / 0.25), np.arange(y_ensure1 - 4, y_ensure1, 0.5)[::-1]]).T
            ensure_path2 = np.vstack(
                [np.arange(x_ensure2 , x_ensure2 + 2, 0.25)[::-1], np.repeat(y_ensure2, 2 / 0.25)]).T
            park_path = self.plan_park_down_right(x_ensure2, y_ensure2)

        elif -math.atan2(0,-1) <= computed_angle <= math.atan2(0,1):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 - self.d - self.w
            y_ensure1 = y_ensure2 - self.l - self.s
            ensure_path1 = np.vstack(
                [np.repeat(x_ensure1, 2 / 0.25), np.arange(y_ensure1 - 4, y_ensure1, 0.5)[::-1]]).T
            ensure_path2 = np.vstack(
                [np.arange(x_ensure2 - 2, x_ensure2, 0.25)[::-1], np.repeat(y_ensure2, 2 / 0.25)]).T
            park_path = self.plan_park_down_left(x_ensure2, y_ensure2)

        elif math.atan2(0,1) < computed_angle <=math.atan2(0,-1) and not left_lane:
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 - self.d - self.w
            y_ensure1 = y_ensure2 + self.l + self.s
            ensure_path1 = np.vstack([np.repeat(x_ensure1, 2 / 0.25), np.arange(y_ensure1, y_ensure1 + 4, 0.5)]).T
            ensure_path2 = np.vstack(
                [np.arange(x_ensure2 - 2, x_ensure2, 0.25)[::-1], np.repeat(y_ensure2, 2 / 0.25)]).T
            park_path = self.plan_park_up_left(x_ensure2, y_ensure2)

        elif math.atan2(0,1) < computed_angle <=math.atan2(0,-1):
            x_ensure2 = gx
            y_ensure2 = gy
            x_ensure1 = x_ensure2 + self.d + self.w
            y_ensure1 = y_ensure2 + self.l + self.s
            ensure_path1 = np.vstack([np.repeat(x_ensure1, 2 / 0.25), np.arange(y_ensure1, y_ensure1 + 4, 0.5)]).T
            ensure_path2 = np.vstack(
                [np.arange(x_ensure2, x_ensure2 + 2, 0.25)[::-1], np.repeat(y_ensure2, 2 / 0.25)]).T
            park_path = self.plan_park_up_right(x_ensure2, y_ensure2)

        return np.array([x_ensure1, y_ensure1]), park_path, ensure_path1, ensure_path2

    def plan_park_up_right(self, x1, y1):
        x0 = x1 + self.d + self.w
        y0 = y1 + 4
        print('Called up right')

        curve_x = np.array([])
        curve_y = np.array([])
        y = np.arange(y0,y1-0.25,-0.25)

        r = np.sqrt((y1-y0)**2)
        x = x0 + np.sqrt(r**2 - (y-y0)**2) - r

        curve_x = np.append(curve_x, x)
        curve_y = np.append(curve_y, y)

        park_path = np.vstack([curve_x, curve_y]).T
        return park_path

    def plan_park_up_left(self, x1, y1):
            x0 = x1 - self.d - self.w
            y0 = y1 + 4
            print('Called up left')

            curve_x = np.array([])
            curve_y = np.array([])
            y = np.arange(y0,y1-0.25,-0.25)

            r = np.sqrt((y1-y0)**2)
            x = x0 - np.sqrt(r**2 - (y-y0)**2) + r

            curve_x = np.append(curve_x, x)
            curve_y = np.append(curve_y, y)

            park_path = np.vstack([curve_x, curve_y]).T
            return park_path


    def plan_park_down_right(self, x1,y1):
        x0 = x1 + self.d + self.w
        y0 = y1 - 4
        print('Called down right')

        curve_x = np.array([])
        curve_y = np.array([])
        y = np.arange(y0,y1+0.25,0.25)

        r = np.sqrt((y1-y0)**2)
        x = x0 + np.sqrt(r**2 - (y-y0)**2) - r

        curve_x = np.append(curve_x, x)
        curve_y = np.append(curve_y, y)

        park_path = np.vstack([curve_x, curve_y]).T
        return park_path


    def plan_park_down_left(self, x1,y1):
        x0 = x1 - self.d - self.w
        y0 = y1 - 4
        print('Called down left')

        curve_x = np.array([])
        curve_y = np.array([])
        y = np.arange(y0,y1+0.25,0.25)

        r = np.sqrt((y1-y0)**2)
        x = x0 - np.sqrt(r**2 - (y-y0)**2) + r

        curve_x = np.append(curve_x, x)
        curve_y = np.append(curve_y, y)

        park_path = np.vstack([curve_x, curve_y]).T
        return park_path
import numpy as np
import math
import matplotlib.pyplot as plt
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

class RRTStar:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None
            self.cost = 0.0

    def __init__(self,ox,oy,resolution=2,rr=4, bounds=np.array([0,1000]),
                 max_extend_length=50.0,
                 goal_sample_rate=0.25,
                 max_iter=50000,
                 connect_circle_dist=500.0,
                 range = 100
                 ):
        self.rr = rr
        self.bounds = bounds
        self.max_extend_length = max_extend_length
        self.resolution = resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.connect_circle_dist = connect_circle_dist
        self.obstacle_map = None
        self.calc_obstacle_map(ox, oy)
        self.range = range
        np.random.seed(0)


    def plan(self,sx,sy,gx,gy):
        """Plans the path from start to goal while avoiding obstacles"""
        self.start = self.Node([sx,sy])
        self.goal = self.Node([gx, gy])
        self.node_list = [self.start]

        self.x_nodes = [self.start.p[0]]
        self.y_nodes = [self.start.p[1]]

        count = 0
        for i in range(self.max_iter):
            # Create a random node inside the bounded environment
            rnd = self.get_random_node(self.node_list[-1])

            x_nodes, y_nodes, duplicate = self.duplicate(self.x_nodes, self.y_nodes, rnd)
            if duplicate:
                continue
            # Find nearest node
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd, self.max_extend_length)
            # If path between new_node and nearest node is not in collision:
            if not self.collision(new_node, nearest_node):
                near_inds = self.near_nodes_inds(new_node)
                # Connect the new node to the best parent in near_inds
                new_node_updated = self.choose_parent(new_node, near_inds, nearest_node)
                if new_node_updated:
                    # Rewire the nodes in the proximity of new_node if it improves their costs
                    self.rewire(new_node, near_inds)
                    self.node_list.append(new_node_updated)
                else:
                    self.node_list.append(new_node)

        print(len(self.x_nodes))
        last_index, min_cost = self.best_goal_node_index()
        if last_index:
            # print(self.final_path(last_index))
            return self.final_path(last_index), self.path_cost(self.final_path(last_index))
        return None, min_cost

    def duplicate(self,x_list,y_list,node):
        x_array = np.array(x_list)
        condition = np.where(x_array == node.p[0])
        if len(condition[0]) != 0:
            for idx in condition[0]:
                if y_list[idx] == node.p[1]:
                    return x_list, y_list, True
        x_list.append(node.p[0])
        y_list.append(node.p[1])
        return x_list, y_list, False

    def choose_parent(self, new_node, near_inds, nearest_node):
        """Set new_node.parent to the lowest resulting cost parent in near_inds and
        new_node.cost to the corresponding minimal cost
        """
        min_cost = np.inf
        best_near_node = None

        cost = []
        node_poss = []
        for i in near_inds:
            if not self.collision(self.node_list[i], new_node):
                # print('running')
                c = self.new_cost(self.node_list[i], new_node)
                cost.append(c)
                node_poss.append(i)

        if len(cost) == 0:

            return None
        best_near_node = self.node_list[node_poss[cost.index(min(cost))]]
        min_cost = min(cost)


        new_node.cost = min_cost
        new_node.parent = best_near_node
        return new_node

    def rewire(self, new_node, near_inds):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            edge_node.cost = self.new_cost(new_node, near_node)

            collision = self.collision(edge_node, new_node)
            improved_cost = near_node.cost > edge_node.cost

            if not collision and improved_cost:
                near_node.p = edge_node.p
                near_node.cost = edge_node.cost
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """Find the lowest cost node to the goal"""
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            # Has to be in close proximity to the goal
            if self.dist_to_goal(node.p) <= self.max_extend_length:
                # Connection between node and goal needs to be collision free
                if not self.collision(self.goal, node):
                    # The final path length
                    cost = node.cost + self.dist_to_goal(node.p)
                    if node.cost + self.dist_to_goal(node.p) < min_cost:
                        # Found better goal node!
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost

    def near_nodes_inds(self, new_node):
        """Find the nodes in close proximity to new_node"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        dlist = [np.sum(np.square((node.p - new_node.p))) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return near_inds

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        d = np.linalg.norm(from_node.p - to_node.p)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    # RRT required functions

    def collision(self,node1,node2):
        """Check whether the path connecting node1 and node2
        is in collision with anyting from the obstacle_map
        """
        p1 = node2.p
        p2 = node1.p

        if p2[0] < self.min_x:
            return False
        elif p2[1] < self.min_y:
            return False

        d12 = p2 - p1  # the directional vector from p1 to p2
        x_sep = np.int(d12[0])
        y_sep = np.int(d12[1])
        path_x = []
        path_y = []
        path_x.append(np.int(p1[0]))
        path_y.append(np.int(p1[1]))
        for i in range(max(abs(x_sep),abs(y_sep))):
            if abs(x_sep)==abs(y_sep):
                sign = np.sign(np.array([x_sep,y_sep]))
                add_x = path_x[i]+sign[0]
                add_y = path_y[i] + sign[1]

            elif abs(x_sep)>abs(y_sep):
                sign = np.sign(x_sep)
                add_x = path_x[i] + sign
                add_y = path_y[i]

            elif abs(x_sep)<abs(y_sep):
                sign = np.sign(y_sep)
                add_x = path_x[i]
                add_y = path_y[i] + sign

            if self.obstacle_map[add_x][add_y]:
                return True
            path_x.append(add_x)
            path_y.append(add_y)
            x_sep = np.int(p2[0])-path_x[i+1]
            y_sep = np.int(p2[1])-path_y[i+1]
            
        return False  # is not in collision

    def final_path(self, goal_ind):
        """Compute the final path from the goal node to the start node"""
        path = [self.goal.p]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.p[0], node.p[1]])
            node = node.parent
        path.append([node.p[0], node.p[1]])
        return np.array(path)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        new_node = self.Node(to_node.p)
        d,theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.p[0]]
        new_node.path_y = [new_node.p[1]]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.resolution)

        for _ in range(n_expand):
            new_node.p[0] += self.resolution * math.cos(theta)
            new_node.p[1] += self.resolution * math.sin(theta)
            new_node.path_x.append(new_node.p[0])
            new_node.path_y.append(new_node.p[1])

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.resolution:
            new_node.path_x.append(to_node.p[0])
            new_node.path_y.append(to_node.p[1])
            new_node.p[0] = to_node.p[0]
            new_node.p[1] = to_node.p[1]

        new_node.parent = from_node

        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return np.linalg.norm(p - self.goal.p)

    def get_random_node(self,node):
        p = node.p
        l_bound = p - self.range
        u_bound = p + self.range
        if l_bound[0] < self.bounds[0]:
            l_bound[0] = self.bounds[0]
        if l_bound[1] < self.bounds[0]:
            l_bound[1] = self.bounds[0]
        if u_bound[0] > self.bounds[1]:
            u_bound[0] = self.bounds[1]
        if u_bound[1] > self.bounds[1]:
            u_bound[1] = self.bounds[1]
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            rnd = self.Node(np.random.uniform(l_bound,u_bound,2))
        else:
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd

    @staticmethod
    def get_nearest_node(node_list, node):
        """Find the nearest node in node_list to node"""
        dlist = [np.sum(np.square((node.p - n.p))) for n in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    @staticmethod
    def path_cost(path):
        return sum(np.linalg.norm(path[i] - path[i + 1]) for i in range(len(path) - 1))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.p[0] - from_node.p[0]
        dy = to_node.p[1] - from_node.p[1]
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = np.zeros((self.x_width*100, self.y_width*100), dtype=bool)

        for ix in range(self.x_width):
            x = ix * self.resolution + self.min_x
            for iy in range(self.y_width):
                y = iy * self.resolution + self.min_y
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d < self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

class PathPlanning:
    def __init__(self,obstacles):
        self.margin = 5
        #sacale obstacles from env margin to pathplanning margin
        obstacles = obstacles + np.array([self.margin,self.margin])
        obstacles = obstacles[(obstacles[:,0]>=0) & (obstacles[:,1]>=0)]

        self.obs = np.concatenate([np.array([[0,i] for i in range(100+self.margin)]),
                                  np.array([[100+2*self.margin,i] for i in range(100+2*self.margin)]),
                                  np.array([[i,0] for i in range(100+self.margin)]),
                                  np.array([[i,100+2*self.margin] for i in range(100+2*self.margin)]),
                                  obstacles])

        self.ox = [int(item) for item in self.obs[:,0]]
        self.oy = [int(item) for item in self.obs[:,1]]
        self.grid_size = 3
        self.robot_radius = 4
        self.rrt_star = RRTStar(self.ox, self.oy, self.grid_size, self.robot_radius)

    def plan_path(self,sx, sy, gx, gy):
        route,cost = self.rrt_star.plan(sx+self.margin, sy+self.margin, gx+self.margin, gy+self.margin)
        print(route)
        rx = []
        ry = []
        for i in route:
            rx.append(i[0])
            ry.append(i[1])
        rx = np.array(rx)-self.margin+0.5
        ry = np.array(ry)-self.margin+0.5
        path = np.vstack([rx,ry]).T
        return path[::-1]
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

    def lane(self, sx, sy, gx, gy):
        Lane = (gx - 10) % 20 < 10
        return Lane

    def generate_park_scenario(self, sx, sy, gx, gy, path):
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
        print(y)
        print(y0,y1)

        r = np.sqrt((y1-y0)**2)
        x = x0 + np.sqrt(r**2 + (y-y0)**2) - r

        curve_x = np.append(curve_x, x)
        curve_y = np.append(curve_y, y)
        print(curve_x)

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
import cv2
import numpy as np

class Environment:
    def __init__(self,obstacles):
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 80/2
        self.car_width = 40/2
        self.wheel_length = np.int(15/2)
        self.wheel_width = np.int(7)
        self.wheel_positions = np.array([[12,7],[12,-7],[-12,7],[-12,-7]])
        #col= np.random.randint(255,size=1)
        #print(col)
        self.color = (np.random.choice(255, 3))/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        # height and width
        self.background = np.ones((1000 + 20 * self.margin, 1000 + 20 * self.margin, 3))
        self.background[10:1000 + 20 * self.margin:10, :] = np.array([200, 200, 200]) / 255
        self.background[:, 10:1000 + 20 * self.margin:10] = np.array([200, 200, 200]) / 255
        self.place_obstacles(obstacles)
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        count = -840
        color = 0
        for ob in obstacles:
            count += 1
            if count % 120 ==0 and count > 0:
                if count >= 13920:
                    color = 0
                else:
                    color = (np.random.choice(120, 3))/255
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=color
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        # rendered[gel[:,1],gel[:,0]] = np.array([0,0,255])/255

        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 0/255, 0/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered


class Parking1:
    def __init__(self, car_pos, numbers):
        self.end = []
        self.car_obstacle = self.make_car()
        # self.walls = []
        self.walls = [[30, i] for i in range(15, 90)] + \
                     [[10, i] for i in range(15, 90)] + \
                     [[70, i] for i in range(15, 90)] + \
                     [[50, i] for i in range(15, 90)] + \
                     [[90, i] for i in range(15, 90)] + \
                     [[i, 15] for i in range(10, 13)] + \
                     [[i, 90] for i in range(10, 13)] + \
                     [[i, 15] for i in range(27, 33)] + \
                     [[i, 90] for i in range(27, 33)] + \
                     [[i, 15] for i in range(47, 53)] + \
                     [[i, 90] for i in range(47, 53)] + \
                     [[i, 15] for i in range(67, 73)] + \
                     [[i, 90] for i in range(67, 73)] + \
                     [[i, 15] for i in range(87, 90)] + \
                     [[i, 90] for i in range(87, 91)] + \
                     [[i, 5] for i in range(-5, 10)]
                     # [[i,j] for i in range(20,23) for j in range(8,11)]+ \
                     # [[i, j] for i in range(60, 63) for j in range(8, 11)]+\
                     # [[i, j] for i in range(100, 103) for j in range(8, 11)]+ \
                     # [[i, j] for i in range(140, 143) for j in range(8, 11)]+ \
                     # [[i, j] for i in range(180, 183) for j in range(8, 11)]+ \
                     # [[i, j] for i in range(20, 23) for j in range(198, 201)] + \
                     # [[i, j] for i in range(60, 63) for j in range(198, 201)] + \
                     # [[i, j] for i in range(100, 103) for j in range(198, 201)] + \
                     # [[i, j] for i in range(140, 143) for j in range(198, 201)] + \
                     # [[i, j] for i in range(180, 183) for j in range(198, 201)]


            # [[i,90] for i in range(70, 76)]
        # + [[i,20] for i in range(-5,50)]
        # self.walls = [0,100]
        self.obs = np.array(self.walls)
        self.cars = dict()
        temp = 1

        for i in range(18):
            x = 20 + i * 4
            for j in range(8):
                if j % 2 == 0:
                    y = 14 + 10 * j
                else:
                    y = 27 + (j - 1) * 10

                d = {temp: [y, x]}
                self.cars.update(d)
                temp = temp + 1


        pos = np.arange(1,143)
        cars_removed = np.random.choice(pos, numbers, replace=False)
        print(cars_removed)
        self.end.append(self.cars[car_pos])
        self.cars.pop(car_pos)
        for car_pos in cars_removed:
            self.end.append(self.cars[car_pos])
            self.cars.pop(car_pos)


    def generate_obstacles(self):
        for i in self.cars.keys():
            for j in range(len(self.cars[i])):
                obstacle = self.car_obstacle + self.cars[i]
                self.obs = np.append(self.obs, obstacle)
        return self.end, np.array(self.obs).reshape(-1,2)

    def make_car(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-2,2), np.arange(-1,1))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2)

        return car_obstacle
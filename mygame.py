import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS
from PIL import Image

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw
import time



width = 128
height = 128

radius = 7
mass = 1

obs_x = width//2
obs_y =  height -   height/8

new_y_posL = 20 
new_y_posH = height//2+20
car_x = width//2
obs_size_offset = 2#10
car_size_offset = 1#5


pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

show_sensors = False
draw_screen = False

COLLTYPE_CAR = 4
COLLTYPE_OBS = 5
import matplotlib.image as mpimg


class GameState:
	def __init__(self):
		# Global-ish.
		self.crashed = False
		self.steps = 0
		# Physics stuff.
		self.space = pymunk.Space()
		self.space.gravity = pymunk.Vec2d(0., 0.)

		# Create the car.
		# Create the car.
		self.create_car(car_x, height//2, np.pi/2)

		# Record steps.
		self.num_steps = 0

		# Create walls.
		static = [
			pymunk.Segment(
				self.space.static_body,
				(0, 1), (0, height), 1),
			pymunk.Segment(
				self.space.static_body,
				(1, height), (width, height), 1),
			pymunk.Segment(
				self.space.static_body,
				(width-1, height), (width-1, 1), 1),
			pymunk.Segment(
				self.space.static_body,
				(1, 1), (width, 1), 1)
		]

		for s in static:
			s.friction = 1.
			s.group = 1
			s.collision_type = 1
			s.color = THECOLORS['red']
		self.space.add(static)


		# Create some obstacles, semi-randomly.
		self.obstacles = []
		self.obstacles.append(self.create_obstacle(obs_x, obs_y, radius + obs_size_offset))
		# self.obstacles.append(self.create_obstacle(700//n, 200//n, 30//n))
		# self.obstacles.append(self.create_obstacle(600//n, 600//n, 30//n))
		# self.obstacles.append(self.create_obstacle(300//n, 400//n, 30//n))
		# self.obstacles.append(self.create_obstacle(500//n, 200//n, 30//n))
		# self.obstacles.append(self.create_obstacle(100//n, 600//n, 30//n))
		self.h = self.space.add_collision_handler(COLLTYPE_CAR, COLLTYPE_OBS,self.collision_occured,None,None,None)
		print(self.h)
	def create_obstacle(self, x, y, r):
		c_body = pymunk.Body(pymunk.inf, pymunk.inf)
		c_shape = pymunk.Circle(c_body, r)
		c_shape.elasticity = 0.0
		c_body.position = x, y
		c_shape.color = THECOLORS["red"]
		c_shape.collision_type = COLLTYPE_OBS
		self.space.add(c_body, c_shape)
		return c_body

	def create_car(self, x, y, r):
		inertia = pymunk.moment_for_circle(mass, 0, 1, (0, 0))
		self.car_body = pymunk.Body(mass, inertia)
		self.car_body.position = x, y
		self.car_shape = pymunk.Circle(self.car_body, radius+car_size_offset)
		self.car_shape.color = THECOLORS["green"]
		self.car_shape.elasticity = 1.0
		self.car_body.angle = r
		self.driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)

		self.car_shape.collision_type = COLLTYPE_CAR
		# self.car_body.apply_impulse(driving_direction)
		self.space.add(self.car_body, self.car_shape)



	def frame_step(self):
	  

		self.driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
		self.car_body.velocity = 100 * self.driving_direction

		# Update the screen and stuff.
		screen.fill(THECOLORS["black"])
		draw(screen, self.space)
		self.space.step(1./10)
		if draw_screen:
			pygame.display.flip()
		
		
		if self.num_steps == 0:
			pygame.image.save(screen,"screenshot.png")
			print("save png")
		
		x, y = self.car_body.position
		# print("val =",self.crashed)
#
		# readings = self.get_sonar_readings(x, y, self.car_body.angle)

		# Get the current location and the readings there.
		
		# Set the reward.
		# Car crashed when any reading == 1
		# if self.car_is_crashed(readings):
		if (self.crashed):
			self.recover_from_crash()  
		else:
			self.num_steps += 1
	   

	def rgb2gray(self,rgb):
   		return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])




	def recover_from_crash(self):
		"""
		We hit something, so recover.
		"""

		print(self.num_steps)
		sc = pygame.image.load("screenshot.png")
		# print(sc)
		img = Image.open('screenshot.png').convert('L')
		# img = self.rgb2gray(img)
		img.save(str(self.num_steps)+'.png')
		# sc = self.rgb2gray(sc)   
		# pygame.image.save(sc,str(self.num_steps)+'.png')
		self.num_steps=0

		# self.crashed = True
		# while self.crashed:
			# Go backwards.
		self.space.remove(self.car_body, self.car_shape)

		self.create_car(car_x,self.new_ypos(),np.pi/2)

		# self.car_body.velocity = -100 * driving_direction
		# self.crashed = False
		# print("q")
		for i in range(10):
			self.car_body.angle += 0  # Turn a little.
			screen.fill(THECOLORS["grey7"])  # Red is scary!
			draw(screen, self.space)
			self.space.step(1./10)
			if draw_screen:
				pygame.display.flip()
				# clock.tick(30)
			# pygame.image.save(screen,"screenshot.jpg")
		self.crashed = False

	def get_img(self):
		# pygame.image.save(screen,"screenshot.jpg")
		img = Image.open("screenshot.png")
		img = self.rgb2gray(img)   
		return np.array(img)


	def new_xpos(self):
		# return random.randint(car_x,width - 100//n)
		return car_x

	def new_ypos(self):
		return random.randint(new_y_posL,new_y_posH )

	def collision_occured(self,space, arbiter):
		# print(self.num_steps-1)
		# sc = pygame.image.load("screenshot.png")
		# pygame.image.save(sc,str(self.num_steps-1)+'.png')
		# self.num_steps=0
		
		print ("you reached the goal!")
		self.crashed = True
		# self.recover_from_crash()
		return True



if __name__ == "__main__":
	game_state = GameState()
	clock = pygame.time.Clock()
	while True:
		# clock.tick(2)
		# time.sleep(0.1)
		game_state.frame_step()

	# def car_is_crashed(self, readings):
	# 	if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
	# 		print(self.num_steps-1)
	# 		sc = pygame.image.load("screenshot.png")
	# 		pygame.image.save(sc,str(self.num_steps-1)+'.png')
	# 		self.num_steps=0
	# 		return True
	# 	else:
	# 		return False


	# def get_sonar_readings(self, x, y, angle):
	# 	readings = []
	# 	"""
	# 	Instead of using a grid of boolean(ish) sensors, sonar readings
	# 	simply return N "distance" readings, one for each sonar
	# 	we're simulating. The distance is a count of the first non-zero
	# 	reading starting at the object. For instance, if the fifth sensor
	# 	in a sonar "arm" is non-zero, then that arm returns a distance of 5.
	# 	"""
	# 	# Make our arms.
	# 	arm_left = self.make_sonar_arm(x, y)
	# 	arm_middle = arm_left
	# 	arm_right = arm_left
	# 	# print(x,y)
	# 	# Rotate them and get readings.
	# 	readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
	# 	readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
	# 	readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

	# 	if show_sensors:
	# 		pygame.display.update()

	# 	return readings

	# def get_arm_distance(self, arm, x, y, angle, offset):
	# 	# Used to count the distance.
	# 	i = 0

	# 	# Look at each point and see if we've hit something.
	# 	for point in arm:
	# 		i += 1

	# 		# Move the point to the right spot.
	# 		rotated_p = self.get_rotated_point(
	# 			x, y, point[0], point[1], angle + offset
	# 		)

	# 		# Check if we've hit something. Return the current i (distance)
	# 		# if we did.
	# 		if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
	# 				or rotated_p[0] >= width or rotated_p[1] >= height:
	# 			return i  # Sensor is off the screen.
	# 		else:
	# 			obs = screen.get_at(rotated_p)
	# 			if self.get_track_or_not(obs) != 0:
	# 				return i

	# 		if show_sensors:
	# 			pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

	# 	# Return the distance for the arm.
	# 	return i

	# def make_sonar_arm(self, x, y):
	# 	spread = 5  # Default spread.
	# 	distance = 0  # Gap before first sensor.
	# 	arm_points = []
	# 	# Make an arm. We build it flat because we'll rotate it about the
	# 	# center later.
	# 	for i in range(1, 4):
	# 		arm_points.append((distance + x + (spread * i), y))

	# 	return arm_points

	# def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
	# 	# Rotate x_2, y_2 around x_1, y_1 by angle.
	# 	x_change = (x_2 - x_1) * math.cos(radians) + \
	# 		(y_2 - y_1) * math.sin(radians)
	# 	y_change = (y_1 - y_2) * math.cos(radians) - \
	# 		(x_1 - x_2) * math.sin(radians)
	# 	new_x = x_change + x_1
	# 	new_y = height - (y_change + y_1)
	# 	return int(new_x), int(new_y)


	# def get_track_or_not(self, reading):
	# 	if reading == THECOLORS['black']:
	# 		return 0
	# 	else:
	# 		return 1

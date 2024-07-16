from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import pytz

# Initialization, define some constants
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/airplane.bmp'
background = plt.imread(filename)

second_hand_length = 200
second_hand_width = 2
minute_hand_length = 150
minute_hand_width = 6
hour_hand_length = 100
hour_hand_width = 10
center = np.array([256, 256])

def clock_hand_vector(angle, length):
    return np.array([length * np.sin(angle), -length * np.cos(angle)])

# Draw an image background
fig, ax = plt.subplots()

while True:
    # Display background image
    ax.imshow(background)

    # First retrieve the time
    now_time = datetime.now()
    hour = now_time.hour % 12
    minute = now_time.minute
    second = now_time.second
    gmt_hour = now_time.hour + 7 

    # Calculate end points of hour, minute, second
    second_angle = second / 60 * 2 * np.pi
    minute_angle = (minute + second / 60) / 60 * 2 * np.pi
    hour_angle = (hour + minute / 60 + second / 3600) / 12 * 2 * np.pi
    gmt_hour_angle = (gmt_hour + minute / 60 + second / 3600) / 24 * 2 * np.pi

    second_vector = clock_hand_vector(second_angle, second_hand_length)
    minute_vector = clock_hand_vector(minute_angle, minute_hand_length)
    hour_vector = clock_hand_vector(hour_angle, hour_hand_length)
    gmt_hour = clock_hand_vector(gmt_hour_angle, hour_hand_length)

    ax.arrow(center[0], center[1], hour_vector[0], hour_vector[1], head_length=3, linewidth=hour_hand_width, color='black')
    ax.arrow(center[0], center[1], minute_vector[0], minute_vector[1], linewidth=minute_hand_width, color='black')
    ax.arrow(center[0], center[1], second_vector[0], second_vector[1], linewidth=second_hand_width, color='red')
    ax.arrow(center[0], center[1], gmt_hour[0], gmt_hour[1], head_length=3, linewidth=hour_hand_width, color='yellow')

    # Remove the axes
    ax.axis('off')
    
    plt.pause(0.1)
    ax.clear()

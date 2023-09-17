import csv
import math
import os
import uuid

import bpy
import numpy as np

OBJECT_SIZE = 0.8
OBJECT_LOCATION = (0, 0, 0)
CAMERA_LOCATION = (0, 0, 0)
CAMERA_ROTATION = (0, 0, 0)
CAMERA_SPHERE_SAMPLES = 10
CAMERA_SPHERE_RADIUS = 4.0
CAMERA_SPHERE_LOCATION = (0, 0, 0)
CAMERA_IMAGE_SIZE = (56, 56)
IMAGE_FILEPATH = "/tmp/"
CSV_FILENAME = "dataset.csv"

bpy.ops.object.delete()

bpy.ops.mesh.primitive_monkey_add(size=OBJECT_SIZE, location=OBJECT_LOCATION)
monkey_head = bpy.data.objects["Suzanne"]

monkey_head.color = (1, 0, 1)

camera = bpy.data.objects["Camera"]

tracking_constraint = camera.constraints.new(type='TRACK_TO')
tracking_constraint.target = monkey_head
tracking_constraint.track_axis = 'TRACK_NEGATIVE_Z'
tracking_constraint.up_axis = 'UP_Y'

camera.location = CAMERA_LOCATION
camera.rotation_euler = CAMERA_ROTATION

print(f'Camera rotation matrix is: {camera.matrix_world.to_3x3()}')

theta, phi = np.meshgrid(
  np.linspace(0, 2 * math.pi, CAMERA_SPHERE_SAMPLES),
  np.linspace(0, math.pi, CAMERA_SPHERE_SAMPLES),
)

with open(CSV_FILENAME, 'w', newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  
  for theta, phi in zip(theta.flatten(), phi.flatten()):
    x = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.cos(phi)
    y = CAMERA_SPHERE_RADIUS * math.sin(theta) * math.sin(phi)
    z = CAMERA_SPHERE_RADIUS * math.cos(theta)

    print(f'Camera location is: ({x}, {y}, {z})')
    camera.location = (x, y, z)

    bpy.context.scene.render.resolution_x = CAMERA_IMAGE_SIZE[0]
    bpy.context.scene.render.resolution_y = CAMERA_IMAGE_SIZE[1]

    image_filename = uuid.uuid4().hex + '.png'

    bpy.ops.render.render(write_still=True, use_viewport=True)
    bpy.data.images['Render Result'].save_render(
      filepath=os.path.join(IMAGE_FILEPATH, image_filename)
    )
    
    camera_orientation = camera.matrix_world.to_3x3()
    
    row = [image_filename]
    row += [x, y, z]
    row += [i for i in camera_orientation[x] for x in range(3)]
    row += [theta, phi]
    csvwriter.writerows(row)
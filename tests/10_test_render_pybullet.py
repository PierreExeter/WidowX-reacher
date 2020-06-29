import gym
import time
import widowx_env
import pybullet
import pkgutil
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('widowx_reacher-v5')


# egl = pkgutil.get_loader('eglRenderer')
# if (egl):
#   pybullet.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
# else:
#   pybullet.loadPlugin("eglRendererPlugin")


env.reset()

camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]

pitch = -10.0
roll = 0
yaw = 0
upAxisIndex = 2
camDistance = 1
pixelWidth = 3200
pixelHeight = 2000
nearPlane = 0.01
farPlane = 100

fov = 60

viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                        roll, upAxisIndex)
aspect = pixelWidth / pixelHeight
projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
img_arr = pybullet.getCameraImage(pixelWidth,
                                  pixelHeight,
                                  viewMatrix,
                                  projectionMatrix,
                                  shadow=1,
                                  lightDirection=[1, 1, 1],
                                  renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)


w = img_arr[0]  #width of the image, in pixels
h = img_arr[1]  #height of the image, in pixels
rgb = img_arr[2]  #color data RGB
dep = img_arr[3]  #depth data

print('width = %d height = %d' % (w, h))

#note that sending the data to matplotlib is really slow

# #reshape is not needed
print(rgb.shape)

np_img_arr = np.reshape(rgb, (h, w, 4))
np_img_arr = np_img_arr * (1. / 255.)
print(np_img_arr.shape)

#show
# plt.imshow(np_img_arr, interpolation='none', extent=(0, 1600, 0, 1200))
image = plt.imshow(np_img_arr, interpolation='none', animated=False)

plt.axis('off')

# image.set_data(np_img_arr)
# ax.plot([0])
# # # plt.draw()
# # # plt.show()
# # # plt.pause(0.01)
# # # image.draw()
plt.savefig("plot_test.png", bbox_inches='tight', dpi=100)



for t in range(1000):
    # action = env.action_space.sample()  
    action = np.zeros(6)
    obs, reward, done, info = env.step(action) 

    print("timestep: ", t)
    time.sleep(1./30.) 

env.close()

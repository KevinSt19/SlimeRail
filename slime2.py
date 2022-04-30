import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from numpy.fft import fft2, ifft2
import time

# def main

def main():

    # w from image:
    #import image for sense data and clean it up
    
    image = img.imread('2020popdist_background.png')
    
    image = np.round_(image)
    
    w = np.zeros((image.shape[0],image.shape[1]), dtype = int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            w[i][j] = sum(image[i][j])
            if w[i][j] == 4:
                w[i][j] = 200
                
    del image

    # variables

    L = w.shape[0]    # side length of box (100)
    W = w.shape[1]    # width of box (L)
    dep = 150   # initial deposit value (150)
    v0 = 1     # total velocity
    pi = np.pi
    sRange = 10  # sense range of particles (10)
    sAngle = pi/6
    N = 5000

    # initial state

    np.random.seed(int(time.time())) #17

    # Particle positions chosen based on image:
#    x = np.random.rand(N,1)*(625) + 25   # random x values of particles
#    y = np.random.rand(N,1)*(975) + 25   # random y values of particles  

    # Particles start in locations based on image
    
    # I don't know what the fuck is happening, but the move step is converting x to an NxN array
    # Possibly something to do with the slice not being a real array
    # or because the shape of x is different than that of theta?
    
    a,b = np.nonzero(w == 200)
    x = a[0::4].copy()
    y = b[0::4].copy()
        
#    x = x.astype('float64')
#    y = y.astype('float64')
    
    N = np.shape(x)[0]
    
    x = np.reshape(x, (N,1))
    y = np.reshape(y, (N,1))
    
    z = np.zeros((L,W), dtype = int)    # deposited sense array
    
    for i in range(N):                  # set initial sense state
        _x = int(np.round_(x[i]))
        _y = int(np.round_(y[i]))
        z[_x][_y] = dep

    theta = np.random.rand(N,1)*2*pi    # random starting orientations of particles

    # prep figure

    for i in range(2000):       # number of time steps
    
        # sense
        for j in range(N):
            _x = int(np.round_(x[j])+sRange*np.cos(theta[j])) % L
            _y = int(np.round_(y[j])+sRange*np.sin(theta[j])) % W
            front = z[_x][_y] + w[_x][_y]

            _x = int(np.round_(x[j])+sRange*np.cos(theta[j]+sAngle)) % L
            _y = int(np.round_(y[j])+sRange*np.sin(theta[j]+sAngle)) % W
            left = z[_x][_y] + w[_x][_y]

            _x = int(np.round_(x[j])+sRange*np.cos(theta[j]-sAngle)) % L
            _y = int(np.round_(y[j])+sRange*np.sin(theta[j]-sAngle)) % W
            right = z[_x][_y] + w[_x][_y]

            # rotate

            if (front == right) and (left == front):
                #theta[j] += np.random.choice([pi/4, -1*pi/4])
                theta[j] += (np.random.rand(1,1)[0][0] * pi/2) * np.random.choice([-1,1])
            elif (right > front) and (right > left):
                theta[j] -= pi/4
            elif (left > front) and (left > right):
                theta[j] += pi/4
            elif (left > front) and (left == right):
                theta[j] += np.random.choice([pi/4, -1*pi/4])
            if (x[j] < 25) or (x[j] > L-25) or (y[j] < 25) or (y[j] > W-25):
                theta[j] = 3*pi/2 + np.arctan2(((L/2)-x[j]),(y[j]-(W/2)))
                
        theta = theta % (2*pi)

        # move
       
        x = x + (v0 * np.cos(theta))
        x = x % L
        y = y + (v0 * np.sin(theta))
        y = y % W

        # diffuse
        
        kernel = [[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]]  # 3x3 gaussian kernel
        kl = 3
        ks = int((kl-1)/2)
        zPrime = ifft2( fft2(z, s=(L+kl-1,W+kl-1)) * fft2(kernel, s=(L+kl-1,W+kl-1)))
        zPrime = zPrime[ks:-1*ks, ks:-1*ks]
        z = zPrime.astype(int)

        
        # deposit
        _x = x.astype(int)
        _y = y.astype(int)
        z[_x,_y] = dep
        
 #       for j in range(N):                  # update sense state
 #           _x = int(np.round_(x[j]))
 #           _y = int(np.round_(y[j]))
 #           z[_x][_y] = dep

        # decay
        
        #z = z*0.99

        # plot z
        plt.gray()
        plt.cla()
        plt.imshow(z+w)
        print(i)
        plt.pause(0.0001)
          
    plt.show()

if __name__ == "__main__":
    main()
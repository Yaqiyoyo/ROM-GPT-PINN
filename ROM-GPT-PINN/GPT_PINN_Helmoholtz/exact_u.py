import numpy as np
import scipy.special as sp
import math

R = 0.25  # Radius of circle (S)
freq  = 0.3  # 0.9 GHz = 900 Mhz
eps1_r = 1.0    # Real part of electric permittivity outside the disk
eps1_i = 0.0    # Imaginary part of electric permittivity outside the disk

# Real part of electric permittivity inside the disk
eps2_i = 0.0     # Imaginary part of electric permittivity inside the disk

mu_r = 1.0    

Z_0 = 1.0    # In vacuum (the incident plane wave is injected in vacuum)

v_0_1 = 0.3  # Velocity 1 (outside the disk) 


# Definition of Bessel and Hankel functions

def bessel_function(x, order):
    return sp.jv(order, x)

def bessel_derivative(x, order):
    return sp.jvp(order,x,n=1)

def hankel_first_kind(x, order):
    return sp.hankel1(order, x)

def hankel_first_kind_derivative(x, order):
    return sp.h1vp(order,x,n=1)


# Defintion of the field outside the disk

def u_e(r,theta, kx1, kx2):
    kx1 = kx1
    kx2 = kx2
    N = 100
    u_e = complex(0.0,0.0)
    
    for n in range(1, N + 1):
        
        i = complex(0,1)
        m = float(n)
       
        An1 = mu_r*kx2*bessel_derivative(-kx2*R,m)*bessel_function(-kx1*R,m)-kx1*bessel_function(-kx2*R,m)*bessel_derivative(-kx1*R,m)
        An2 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx2*R,m)-mu_r*kx2*bessel_derivative(-kx2*R,m)*hankel_first_kind(-kx1*R,m)
        An  = An1/An2
        
        #print("An:", An)

        u_e = u_e + (i**m)*(bessel_function(-kx1*r,m)+ An*hankel_first_kind(-kx1*r,m))*np.cos(m*theta) 

        #print("u_e:", u_e )    
   
    
    A01 = mu_r*kx2*bessel_derivative(-kx2*R,0)*bessel_function(-kx1*R,0)-kx1*bessel_function(-kx2*R,0)*bessel_derivative(-kx1*R,0)
    A02 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx2*R,0)-mu_r*kx2*bessel_derivative(-kx2*R,0)*hankel_first_kind(-kx1*R,0)
    A0  = A01/A02

    u_e = bessel_function(-kx1*r,0) + A0*hankel_first_kind(-kx1*r,0) + 2*u_e

    return u_e 

# Definition of the field inside the disk

def u_i(r,theta, kx1, kx2):
    kx1 = kx1
    kx2 = kx2
    N = 100
    u_i = complex(0.0,0.0)
    
    for n in range(1, N + 1):
        
        i = complex(0,1)
        m = float(n)
        
        Bn1 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx1*R,m)-kx1*hankel_first_kind(-kx1*R,m)*bessel_derivative(-kx1*R,m)
        Bn2 = kx1*hankel_first_kind_derivative(-kx1*R,m)*bessel_function(-kx2*R,m)-mu_r*kx2*bessel_derivative(-kx2*R,m)*hankel_first_kind(-kx1*R,m)
        Bn  = Bn1/Bn2
      
        #print("Bn:", Bn)

        u_i = u_i + (i**m)*Bn*bessel_function(-kx2*r,m)*np.cos(m*theta) 
      
        #print("u_i:", u_i)

    B01 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx1*R,0)-kx1*hankel_first_kind(-kx1*R,0)*bessel_derivative(-kx1*R,0)
    B02 = kx1*hankel_first_kind_derivative(-kx1*R,0)*bessel_function(-kx2*R,0)-mu_r*kx2*bessel_derivative(-kx2*R,0)*hankel_first_kind(-kx1*R,0)
    B0  = B01/B02
    
    
    u_i = B0*bessel_function(-kx2*r,0) + 2*u_i
    
    return u_i

# Global field

def u(r, theta, eps2_r):
    eps2_r = eps2_r
    v_0_2 = v_0_1 / np.sqrt(mu_r * eps2_r)  # Velocity 2 (inside the disk) if you change eps2_r, you must also change v_0_2 (in case eps2_r = 4.0, we have v_0_2 = 0.15)
    omk = (2.0 * math.pi * freq) / v_0_1  # Pulsation
    omk2 = (2.0 * math.pi * freq) / v_0_2  # Pulsation
    kx1 = omk  # outside the disk
    kx2 = omk2  # inside the disk

    u_i_values = u_i(r, theta, kx1, kx2)
    u_e_values = u_e(r, theta, kx1, kx2)
   
    return np.where(r <= R, u_i_values, u_e_values)
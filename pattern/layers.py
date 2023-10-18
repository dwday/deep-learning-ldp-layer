# -*- coding: utf-8 -*-
"""
Local derivative patterns implementation using custom layer definitions

Akgun, Devrim. "TensorFlow based deep learning layer for Local Derivative Patterns." 
Software Impacts 14 (2022): 100452.
https://www.sciencedirect.com/science/article/pii/S2665963822001361
"""

import tensorflow as tf
from keras import layers


class LDP(layers.Layer):
    
    # Initialize variables
    def __init__(self,mode='single',alpha='0',**kwargs): 
        self.mode=mode
        self.alpha=alpha        
        super(LDP,self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        super(LDP,self).build(input_shape)
    
   
    def call(self, x):              
        if x.shape[1]==None:
            return x
        if  self.mode=='single'  and  self.alpha=='0':
            z=tf_ldp0(x)            
        elif self.mode=='single' and  self.alpha=='45':
            z=tf_ldp45(x)
        elif self.mode=='single' and  self.alpha=='90':
            z=tf_ldp90(x)
        elif self.mode=='single' and  self.alpha=='135':
            z=tf_ldp135(x)  
        elif self.mode=='multi':
            # compute all features
            z1=tf_ldp0(x)
            z2=tf_ldp45(x)
            z3=tf_ldp90(x)
            z4=tf_ldp135(x) 
            # concatenate all features
            z=tf.concat([z1,z2,z3,z4],axis=1)     
            
        elif self.mode=='mean':   
            # compute all features
            z1=tf_ldp0(x)
            z2=tf_ldp45(x)
            z3=tf_ldp90(x)
            z4=tf_ldp135(x) 
            
            # sum featureas
            z=tf.add(z1,z2)
            z=tf.add(z,z3)
            z=tf.add(z,z4)
            # compute mean
            z=tf.divide(z,tf.constant(4.0))           
            
        else:
            print('Warning: wrong input parameters. Defaults to LDP(alpha=0)')
            z=tf_ldp0(x)                 
            
        return tf.cast(z,dtype=tf.float32) 
       
    def compute_output_shape(self,input_shape):
        assert isinstance(input_shape,list)
        b=input_shape
        # height=b[1]
        # width=b[2]
        out1=(b[0],b[1],b[2],b[3])
        return out1  

#******************************************************************************
def tf_ldp0(Im):
    
    # -- PADDING
    # Im=tf.pad(Im, tf.constant([[0,0],[1,2],[1, 2],[0,0]]) )  
    Im=tf.pad(Im, tf.constant([[0,0],[2,1],[2, 1],[0,0]]) )

    #------------------------------------------------------
    M=Im.shape[1]
    N=Im.shape[2]   
    
    #select elements within the neigbourhood    
    y10=Im[:,1:M-2, 0:N-3,:]
    y11=Im[:,1:M-2, 1:N-2,:]
    y12=Im[:,1:M-2, 2:N-1,:]
    y13=Im[:,1:M-2, 3:N  ,:]
    #
    y20=Im[:,2:M-1, 0:N-3,:]
    y21=Im[:,2:M-1, 1:N-2,:]
    y22=Im[:,2:M-1, 2:N-1,:]       
    y23=Im[:,2:M-1, 3:N  ,:]       
    #
    y30=Im[:,3:M,   0:N-3,:]
    y31=Im[:,3:M,   1:N-2,:]
    y32=Im[:,3:M,   2:N-1,:]       
    y33=Im[:,3:M,   3:N  ,:]   
   
    z=y11       
    pc=tf.subtract(y21,y22)
    #00--------------------------------
    # cond_true  = tf.zeros(tf.shape(z),dtype=tf.float32)    
    cond_true  = tf.multiply(z,0)     
    cond_false = tf.math.add( cond_true,tf.constant(128.0,dtype=tf.float32) )
    d=tf.subtract(y10,y11)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)                 
    z= tf.where(g,cond_true,cond_false)  

    #01--------------------------------
    cond_true = z       
    d=tf.subtract(y11,y12)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(64.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #02--------------------------------
    cond_true = z      
    d=tf.subtract(y12,y13)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add(z,tf.constant(32.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #10-------------------------------
    cond_true = z      
    d=tf.subtract(y20,y21)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)     
    cond_false = tf.add( z,tf.constant(1.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #12-------------------------------
    cond_true = z       
    d=tf.subtract(y22,y23)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(16.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #20-------------------------------
    cond_true = z       
    d=tf.subtract(y30,y31)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(2.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #21-------------------------------
    cond_true = z       
    d=tf.subtract(y31,y32)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(4.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #22-------------------------------
    cond_true = z        
    d=tf.subtract(y32,y33)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(8.0,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    z=tf.divide(z,255.0 )   
    return z

#******************************************************************************
def tf_ldp45(Im):    

    # -- PADDING
    # Im=tf.pad(Im, tf.constant([[0,0],[1,2],[1, 2],[0,0]]) )
    Im=tf.pad(Im, tf.constant([[0,0],[2,1],[2, 1],[0,0]]) )

    #--------------------------------------------------------------------------
    M=Im.shape[1]
    N=Im.shape[2]   

    #select elements within the 3x3 neigbourhood
    # y00=Im[:,0:M-3, 0:N-3,:]
    y01=Im[:,0:M-3, 1:N-2,:]
    y02=Im[:,0:M-3, 2:N-1,:]
    y03=Im[:,0:M-3, 3:N  ,:]
    #     
    y10=Im[:,1:M-2, 0:N-3,:]
    y11=Im[:,1:M-2, 1:N-2,:]
    y12=Im[:,1:M-2, 2:N-1,:]
    y13=Im[:,1:M-2, 3:N  ,:]
    #
    y20=Im[:,2:M-1, 0:N-3,:]
    y21=Im[:,2:M-1, 1:N-2,:]
    y22=Im[:,2:M-1, 2:N-1,:]       
    y23=Im[:,2:M-1, 3:N  ,:]       
    #
    y30=Im[:,3:M,   0:N-3,:]
    y31=Im[:,3:M,   1:N-2,:]
    y32=Im[:,3:M,   2:N-1,:]       
    # y33=Im[:,3:M,   3:N  ,:]   
   
    z=y11       
    pc=tf.subtract(y21,y12)
    #00-------------------------------- 
    # cond_true = tf.zeros(tf.shape(z),dtype=tf.uint8)    
    cond_true  = tf.multiply(z,0) 
    cond_false = tf.add( cond_true,tf.constant(128,dtype=tf.float32) )
    d=tf.subtract(y10,y01)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)                       
    z=tf.where(g,cond_true,cond_false)  
    
    #01--------------------------------
    cond_true = z       
    d=tf.subtract(y11,y02)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(64,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #02--------------------------------
    cond_true = z      
    d=tf.subtract(y12,y03)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add(z,tf.constant(32,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #10-------------------------------
    cond_true = z      
    d=tf.subtract(y20,y11)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)     
    cond_false = tf.add( z,tf.constant(1,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #12-------------------------------
    cond_true = z       
    d=tf.subtract(y22,y13)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(16,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #20-------------------------------
    cond_true = z       
    d=tf.subtract(y30,y21)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(2,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #21-------------------------------
    cond_true = z       
    d=tf.subtract(y31,y22)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(4,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #22-------------------------------
    cond_true = z        
    d=tf.subtract(y32,y23)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(8,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)   
    # z=tf.cast(z,dtype=tf.float32)/255.0
    z=tf.divide(z,255.0 ) 
    return z
#------------------------------------------------------------------------------
def tf_ldp90(Im):

    # -- PADDING
    # Im=tf.pad(Im, tf.constant([[0,0],[1,2],[1, 2],[0,0]]) )
    Im=tf.pad(Im, tf.constant([[0,0],[2,1],[2, 1],[0,0]]) )

    #--------------------------------------------------------------------------
    M=Im.shape[1]
    N=Im.shape[2]   
  
    #select elements within the neigbourhood
    y00=Im[:,0:M-3, 0:N-3]
    y01=Im[:,0:M-3, 1:N-2]
    y02=Im[:,0:M-3, 2:N-1]
    # y03=Im[:,0:M-3, 3:N  ]
    #     
    y10=Im[:,1:M-2, 0:N-3]
    y11=Im[:,1:M-2, 1:N-2]
    y12=Im[:,1:M-2, 2:N-1]
    # y13=Im[:,1:M-2, 3:N  ]
    #
    y20=Im[:,2:M-1, 0:N-3]
    y21=Im[:,2:M-1, 1:N-2]
    y22=Im[:,2:M-1, 2:N-1]       
    # y23=Im[:,2:M-1, 3:N  ]       
    #
    y30=Im[:,3:M,   0:N-3]
    y31=Im[:,3:M,   1:N-2]
    y32=Im[:,3:M,   2:N-1]       
    # y33=Im[:,3:M,   3:N  ]   
   
    z=y11       
    pc=tf.subtract(y21,y11)
    #00--------------------------------
    # cond_true = tf.zeros(tf.shape(z),dtype=tf.uint8)   
    cond_true  = tf.multiply(z,0) 

    cond_false = tf.add( cond_true,tf.constant(128,dtype=tf.float32) )
    d=tf.subtract(y10,y00)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)                       
    z=tf.where(g,cond_true,cond_false)  
    
    #01--------------------------------
    cond_true = z       
    d=tf.subtract(y11,y01)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(64,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #02--------------------------------
    cond_true = z      
    d=tf.subtract(y12,y02)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add(z,tf.constant(32,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #10-------------------------------
    cond_true = z      
    d=tf.subtract(y20,y10)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)     
    cond_false = tf.add( z,tf.constant(1,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #12-------------------------------
    cond_true = z       
    d=tf.subtract(y22,y12)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(16,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #20-------------------------------
    cond_true = z       
    d=tf.subtract(y30,y20)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(2,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #21-------------------------------
    cond_true = z       
    d=tf.subtract(y31,y21)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(4,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #22-------------------------------
    cond_true = z        
    d=tf.subtract(y32,y22)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(8,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    # z=tf.cast(z,dtype=tf.float32)/255.0   
    z=tf.divide(z,255.0 ) 
    return z

#------------------------------------------------------------------------------
def tf_ldp135(Im):

    # -- PADDING
    # Im=tf.pad(Im, tf.constant([[0,0],[1,2],[1, 2],[0,0]]) )
    Im=tf.pad(Im, tf.constant([[0,0],[2,1],[2, 1],[0,0]]) )

    #--------------------------------------------------------------------------
    M=Im.shape[1]
    N=Im.shape[2]   


    #select elements within the neigbourhood
    y00=Im[:,0:M-3, 0:N-3,:] 
    y01=Im[:,0:M-3, 1:N-2,:] 
    y02=Im[:,0:M-3, 2:N-1,:] 
    # y03=Im[:,0:M-3, 3:N  ,:] 
    #     
    y10=Im[:,1:M-2, 0:N-3,:] 
    y11=Im[:,1:M-2, 1:N-2,:] 
    y12=Im[:,1:M-2, 2:N-1,:] 
    y13=Im[:,1:M-2, 3:N  ,:] 
    #
    y20=Im[:,2:M-1, 0:N-3,:] 
    y21=Im[:,2:M-1, 1:N-2,:] 
    y22=Im[:,2:M-1, 2:N-1,:]        
    y23=Im[:,2:M-1, 3:N  ,:]       
    #
    # y30=Im[:,3:M,   0:N-3,:] 
    y31=Im[:,3:M,   1:N-2,:] 
    y32=Im[:,3:M,   2:N-1,:]        
    y33=Im[:,3:M,   3:N  ,:]   
    #---------------------
    z=y11       
    pc=tf.subtract(y22,y11)
    #00--------------------------------
    # cond_true = tf.zeros(tf.shape(z),dtype=tf.uint8)     
    cond_true  = tf.multiply(z,0) 

    cond_false = tf.add( cond_true,tf.constant(128,dtype=tf.float32) )
    d=tf.subtract(y11,y00)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)                       
    z=tf.where(g,cond_true,cond_false)  
    
    #01--------------------------------
    cond_true = z       
    d=tf.subtract(y12,y01)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(64,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #02--------------------------------
    cond_true = z      
    d=tf.subtract(y13,y02)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add(z,tf.constant(32,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #10-------------------------------
    cond_true = z      
    d=tf.subtract(y21,y10)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0)     
    cond_false = tf.add( z,tf.constant(1,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #12-------------------------------
    cond_true = z       
    d=tf.subtract(y23,y12)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(16,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    #20-------------------------------
    cond_true = z       
    d=tf.subtract(y31,y20)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(2,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #21-------------------------------
    cond_true = z       
    d=tf.subtract(y32,y21)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(4,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false) 
    #22-------------------------------
    cond_true = z        
    d=tf.subtract(y33,y22)
    d=tf.multiply(d,pc)
    g=tf.greater_equal(d,0) 
    cond_false = tf.add( z,tf.constant(8,dtype=tf.float32) )                     
    z=tf.where(g,cond_true,cond_false)  
    
    # z=tf.cast(z,dtype=tf.float32)/255.0
    z=tf.divide(z,255.0 ) 
    return z


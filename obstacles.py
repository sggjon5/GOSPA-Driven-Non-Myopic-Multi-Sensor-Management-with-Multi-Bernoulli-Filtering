# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:06:01 2024

@author: sggjone5

"""
import numpy as np



class obstacle_one():
    """
    cloud obstacle
    
    - groundtruth not affected
    - measurements blocked
    
    takes in a circle centre as an (x,y) cooridnate
    takes in an int radius for the cloud size
    """
    
    def __init__(self, location_centre, radius):
        self.location_centre = location_centre
        self.radius = radius
        self.name = 'blockade'
        self.blocks_measurements = False
        self.blocks_groundtruth = False
        self.blocks_sensor = True
    
    
    
class obstacle_two():
    
    def __init__(self, A, B, C, D):
        
        # where the rectangle is defined as
        # A        B
        # 
        # D        C
        
        # rectangle can be at an angle
        
        self.name = "rectangle blockade"
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        self.blocks_measurements = False
        self.blocks_groundtruth = False
        self.blocks_sensor = True
        
        
    def check_if_point_inside(self, coordinate):
        
        """
        Takes in a coorinate and checks whether it is inside our outside the rectangle
        used the answer by hkBattousai @ 
        https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
        
        returns True if inside of the area
        returns False if outside of the area
        """
        
        x = coordinate[0]
        y = coordinate[1]
        
        x1 = self.A[0]
        x2 = self.B[0]
        x3 = self.C[0]
        x4 = self.D[0]
        
        y1 = self.A[1]
        y2 = self.B[1]
        y3 = self.C[1]
        y4 = self.D[1]
        
        #calculate edge lengths
        edge1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        edge2 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        edge3 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
        edge4 = np.sqrt((x4 - x1)**2 + (y4 - y1)**2)
        
        # calculate lengths of line segments
        line_length1 = np.sqrt((x1 - x)**2 + (y1 - y)**2)
        line_length2 = np.sqrt((x2 - x)**2 + (y2 - y)**2)
        line_length3 = np.sqrt((x3 - x)**2 +  (y3 - y)**2)
        line_length4 = np.sqrt((x4 - x)**2 + (y4 - y)**2)
        
        # calculate areas using Heron's Forumla
        rectangle_area = edge1*edge2
        
        u1 = (edge1 + line_length1 + line_length2)/2
        u2 = (edge2 + line_length2 + line_length3)/2
        u3 = (edge3 + line_length3 + line_length4)/2
        u4 = (edge4 + line_length4 + line_length1)/2
        
        
        # calculate the areas of the four triangles
        area1 = np.sqrt(u1*(u1 - edge1)*(u1 - line_length1)*(u1 - line_length2))
        area2 = np.sqrt(u2*(u2 - edge2)*(u2 - line_length2)*(u2 - line_length3))
        area3 = np.sqrt(u3*(u3 - edge3)*(u3 - line_length3)*(u3 - line_length4))
        area4 = np.sqrt(u4*(u4 - edge4)*(u4 - line_length4)*(u4 - line_length1))
        
        total_triangle_area = area1 + area2 + area3 + area4
        
        if total_triangle_area > rectangle_area:
            return False
        else:
            return True
        
        
class obstacle_three():

    def __init__(self, bottom_left, top_right):
        self.name = "flat_rectangle_blockade"
        self.bl = bottom_left
        self.tr = top_right
        
        self.A = np.array([[self.bl[0][0]],[self.tr[1][0]]])
        self.B = self.tr
        self.C = np.array([[self.tr[0][0]],[self.bl[1][0]]])
        self.D = self.bl
    
    def check_if_point_inside(self, point):
        """
        uses a top right and bottom left coordinates in the form of (x,y)
        from the self, do not need to pass these as a method that checks whether the point being assessed is inside which is 
        also passed as an (x,y) coordinate pair
        
        returns True if inside of the area
        returns False if outside of the area
        """
        if (point[0] > self.bl[0] and point[0] < self.tr[0] and point[1] > self.bl[1] and point[1] < self.tr[1]):
            return True
        else:
            return False
            
        
        
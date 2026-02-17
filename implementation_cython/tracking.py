# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 21:09:24 2026

@author: deniz
"""

import numpy as np
import math

class ObstacleTracker:
    def __init__(self, max_disappeared=5):
        self.nextObjectID = 0
        self.objects = {}  # ID -> Centroid (x, y)
        self.velocities = {} # ID -> Vector (vx, vy)
        self.disappeared = {} # ID -> compteur de frames sans detection
        self.max_disappeared = max_disappeared

    def update(self, rects):
        # rects = liste de [x1, y1, x2, y2] venant de tes bounding boxes
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects, self.velocities

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Si aucun objet n'est tracké, on les enregistre tous
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # ICI: Logique d'appariement (Matching)
            # On compare les distances entre les anciens et nouveaux centres
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = [] # Matrice de distance
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                
                objectID = objectIDs[row]
                self.update_object(objectID, inputCentroids[col])
                
                usedRows.add(row)
                usedCols.add(col)

            # Gestion des disparitions et nouveaux objets
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects, self.velocities

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.velocities[self.nextObjectID] = (0, 0) # Vitesse nulle au début
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.velocities[objectID]
        del self.disappeared[objectID]

    def update_object(self, objectID, new_centroid):
        # Calcul du vecteur vitesse (Delta position)
        old_centroid = self.objects[objectID]
        vx = new_centroid[0] - old_centroid[0]
        vy = new_centroid[1] - old_centroid[1]
        
        # Lissage simple de la vitesse (moyenne mobile)
        old_vx, old_vy = self.velocities[objectID]
        self.velocities[objectID] = (0.7 * old_vx + 0.3 * vx, 0.7 * old_vy + 0.3 * vy)
        
        self.objects[objectID] = new_centroid
        self.disappeared[objectID] = 0
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

    def __init__(self, maxDisappearance, maxDistance):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappearance = maxDisappearance

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

        # register a new object ID
    def register(self, centroID):
        # store the object
        self.objects[self.nextObjectID] = centroID
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    # removes the existing object ID
    def deregister(self, centroID):
        del self.objects[centroID]
        del self.disappeared[centroID]

    # updates the existing object ID
    def update(self, rects):
        # check for empty boxes
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if reached the maximum number of consecutive frames where the object has been marked as
                # missing, deregister
                if self.disappeared[objectID] > self.maxDisappearance:
                    self.deregister(objectID)

            # return if there are no tracking info
            return self.objects

        # initialize array of input object ID in the current frame
        inputCentroIDs = np.zeros((len(rects), 2), "int")

        # loop over the bounding box rectangles
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            # use the coordinates
            c_x = int((start_x + end_x) // 2)
            c_y = int((start_y + end_y) // 2)
            inputCentroIDs[i] = (c_x, c_y)

        # register new centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroIDs)):
                self.register(inputCentroIDs[i])

        # otherwise replace the current existing tracking centroids
        else:
            # grab set of object IDs to the corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroIDs = list(self.objects.values())

            # compute the distance between the object centroids and input centroids, match an input centroid to
            # an existing object centroid
            D = dist.cdist(np.array(objectCentroIDs), inputCentroIDs)

            # 1 - find the smallest value of each row
            # 2 - sort the row index based on their minimum values
            rows = D.min(1).argsort()

            # find the smallest value in each column then sort using the previous computed row index list
            cols = D.argmin(1)[rows]

            # keep track of which rows and column indexes has been examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of (row, column) tuples
            for (row,col) in zip(rows, cols):
                # if the row or column has been examined, ignore
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than the maximum distance, do not associate the two centroids
                if D[row, col] > self.maxDistance:
                    continue

                # grab the object ID for the current row and set as a new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroIDs[col]
                self.disappeared[objectID] = 0

                # indicate that each row and columns are indexed respectively
                usedCols.add(col)
                usedRows.add(row)

            # compute both row and column index that has not yet been examined
            unused_rows = set(range(0, D.shape[0])).difference(usedRows)
            unused_cols = set(range(0, D.shape[1])).difference(usedCols)

            # check if some object centroids is equal or greater than the number of input centroids that may have disappeared
            if D.shape[0] >= D.shape[1]:

                # loop unused row indexes
                for row in unused_rows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive frames the object has been marked "disappeared" to register the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # if the number of input centroids is greater than the number of existing object centroids to register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(inputCentroIDs[col])

        # return set of trackable objects
        return self.objects
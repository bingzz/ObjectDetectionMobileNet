class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the objectID and initialize the list of centroids using the current centroid
        self.objectID = objectID
        self.centroid = [centroid]

        # initialize a boolean if the object has already been counted
        self.counted = False

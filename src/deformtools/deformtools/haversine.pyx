from math import radians, cos, sin, asin, sqrt, atan2
cdef float lon1, lon2,lat1,lat2,km,c,dlat,dlon
def haversine(p1, p2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a),sqrt(1-a)) 
    km = 6371 * c
    return km
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Connection to SITL
connection_string = '127.0.0.1:14551' # or 14551 if using the split method
print(f"--- Connecting to: {connection_string} ---")
vehicle = connect(connection_string, wait_ready=True)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobalRelative object containing the latitude/longitude `dNorth` and `dEast` metres from the specified `original_location`.
    """
    from math import pi
    earth_radius = 6378137.0 # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*(pi/180.0)*180.0) # Simple approximation

    # New position in decimal degrees
    newlat = original_location.lat + (dLat * 180.0/pi)
    newlon = original_location.lon + (dLon * 180.0/pi)
    return LocationGlobalRelative(newlat, newlon, original_location.alt)

def arm_and_takeoff(target_alt):
    print("Basic pre-arm checks...")
    while not vehicle.is_armable:
        time.sleep(1)
    print("Taking off!")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed: time.sleep(1)
    vehicle.simple_takeoff(target_alt)
    while True:
        if vehicle.location.global_relative_frame.alt >= target_alt * 0.95:
            break
        time.sleep(1)

try:
    arm_and_takeoff(10)
    
    # Starting point
    start_loc = vehicle.location.global_relative_frame

    # Define Square Points (North, East)
    points = [
        (100, 0),   # 100m North
        (100, 100), # 100m North, 100m East
        (0, 100),   # Back to original Latitude, but still 100m East
        (0, 0)      # Back to start
    ]

    for dN, dE in points:
        target = get_location_metres(start_loc, dN, dE)
        print(f"Moving to: North {dN}m, East {dE}m")
        vehicle.simple_goto(target)
        
        # Wait until we are close to the target
        time.sleep(15) 

    print("Mission Complete. Returning to Land.")
    vehicle.mode = VehicleMode("RTL")

except Exception as e:
    print(f"Error: {e}")
finally:
    vehicle.close()
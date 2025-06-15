def vehicle_routing():
    Distance = input("in miles")
    Speed = input("in miles per hour")

    time = Distance / Speed
    time_minutes = time * 60
    print("Time taken:", time, "hours")
    print("Time taken:", round(time_minutes, 1), "minutes")

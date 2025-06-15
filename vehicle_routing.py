Distance = input("Distance in miles: ")
Speed = input("Speed in miles per hour: ")

time = int(Distance) / int(Speed)
time_minutes = time * 60
print("Time taken in hours:", time)
print("Time taken in minutes:", round(time_minutes, 1))
    


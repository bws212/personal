

def calculate_aqi(data):
    for i in range(len(data)):

    index = ((IHi - ILo) / (BPHi - BPLo)) * (truncated_concentration - BPLo) + ILo
    return round(index)

def main():
    data = {
        "Regular US Urban day": {
            "PM2.5": 11,  # μg m^-3
            "8-hour O3": 35,  # ppb
            "NO2": 30,  # ppb
            "CO": 4  # ppm
        },
        "Wildfire day": {
            "PM2.5": 250,  # μg m^-3
            "8-hour O3": 35,  # ppb
            "NO2": 30,  # ppb
            "CO": 8  # ppm
        },
        "U.S. Urban haze day": {
            "PM2.5": 50,  # μg m^-3
            "8-hour O3": 80,  # ppb
            "NO2": 75,  # ppb
            "CO": 10  # ppm
        },
        "Delhi smog day": {
            "PM2.5": 250,  # μg m^-3
            "8-hour O3": 70,  # ppb
            "NO2": 150,  # ppb
            "CO": 12  # ppm
        }
    }

    for day, values in data.items():
        print(day)
        for pollutant, value in values.items():
            print(f"{pollutant}: {value}")




if __name__ == "__main__":
    main()
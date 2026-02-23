from utils import check_water

# Try confirmed water locations
tests = [
    (16.4945, 80.6740),
    (19.845, 85.478),
    (17.4239, 78.4738)
]

for lat, lon in tests:
    print("\nTesting:", lat, lon)
    result = check_water(lat, lon)
    print("Water detected:", result)

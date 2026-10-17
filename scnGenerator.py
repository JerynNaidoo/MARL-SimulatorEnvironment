import math
import random
from pathlib import Path


def generate_random_scenario_file(
    output_filename: str = "random_conflict.scn",
    num_aircraft: int = 4,
    center_lat: float = 52.0,
    center_lon: float = 4.0,
    spawn_radius_nm: float = 20.0
):
    lines = [
        f"# Automatically Generated BlueSky Scenario: {num_aircraft} Aircraft Convergence\n",
        "# Center Point: ({center_lat}, {center_lon})\n\n"
    ]

    for i in range(num_aircraft):
        ac_id = f"AC{i+1:02d}"
        
        # Pick random angle along the circle perimeter (0 to 360 deg)
        # Or space them evenly with slight noise:
        angle_deg = (360.0 / num_aircraft) * i + random.uniform(-15.0, 15.0)
        angle_rad = math.radians(angle_deg)
        
        # Calculate spawn latitude/longitude offsets
        d_lat = (spawn_radius_nm / 60.0) * math.cos(angle_rad)
        d_lon = (spawn_radius_nm / (60.0 * math.cos(math.radians(center_lat)))) * math.sin(angle_rad)
        
        spawn_lat = center_lat + d_lat
        spawn_lon = center_lon + d_lon
        
        # Heading points inward toward center point: opposite of spawn bearing
        heading = (angle_deg + 180.0) % 360.0
        
        # Destination waypoint is mirrored across the center point
        dest_lat = center_lat - d_lat
        dest_lon = center_lon - d_lon
        
        speed_kts = random.choice([240, 250, 260]) # Slight speed variances
        
        # Write BlueSky stack commands
        lines.append(f"00:00:00.00> CRE {ac_id}, B738, {spawn_lat:.4f}, {spawn_lon:.4f}, {heading:.1f}, 10000, {speed_kts}\n")
        lines.append(f"00:00:00.00> ADDWPT {ac_id}, {dest_lat:.4f}, {dest_lon:.4f}\n\n")

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Generated scenario '{output_path}' with {num_aircraft} aircraft.")

if __name__ == "__main__":
    generate_random_scenario_file("scenarios/conflict_50ac_random.scn", num_aircraft=50)
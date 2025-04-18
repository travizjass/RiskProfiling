import os
import json

# Define folder paths
with_zip_folder = "withzip"
without_zip_folder = "withoutzip"

# Load JSON files from a folder
def load_json_files(folder_path):
    json_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                json_data[filename] = data
    return json_data

# Save updated JSON data back to files
def save_json_files(folder_path, json_data):
    for filename, data in json_data.items():
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

# Load JSON data from both folders
with_zip_data = load_json_files(with_zip_folder)
without_zip_data = load_json_files(without_zip_folder)

# Create a mapping of id to required fields from the with_zip folder
data_mapping = {}
for data in with_zip_data.values():
    if "id" in data:
        record = {}
        
        # Count high-risk drugs
        medications = data.get("details", {}).get("medications", {}).get("records", [])
        record["high_risk_drugs_count"] = sum(1 for med in medications if med.get("is_high_risk_drug", False))
        
        # Count infections
        infections = data.get("details", {}).get("infections", {}).get("records", [])
        record["infections_count"] = len(infections)
        
        # Count wounds
        wounds = data.get("details", {}).get("wounds", {}).get("records", [])
        record["wounds_count"] = len(wounds)
        
        # Find oxygen_saturation value
        oxygen_saturation = None
        encounters = data.get("encounters", {})
        for visit in encounters.values():
            detailed_info = visit.get("detailed_info", {})
            details = detailed_info.get("details", {})
            
            # Check if details is not None before accessing its keys
            if details:
                oxygen_saturation = details.get("oxygen_saturation")
                if oxygen_saturation:
                    break
        record["oxygen_saturation"] = oxygen_saturation
        
        # Add the record to the data_mapping dictionary
        data_mapping[data["id"]] = record

# Debug: Print the data_mapping to verify its contents
print("Data Mapping:")
print(json.dumps(data_mapping, indent=4))

# Update JSON files in the without_zip folder
updated_count = 0
for filename, data in without_zip_data.items():
    if "id" in data:
        key = data["id"]
        if key in data_mapping:
            data.update(data_mapping[key])
            updated_count += 1

# Save the updated JSON files back to the without_zip folder
save_json_files(without_zip_folder, without_zip_data)

print(f"Data has been successfully updated for {updated_count} files!")
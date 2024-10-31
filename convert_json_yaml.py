import json
import yaml

print("Loading JSON file")
#Load JSON
with open("planes dataset/planesnet.json", "r") as json_file:
    json_data = json.load(json_file)

print("Converting JSON to YAML")
#Convert JSON to YAML
yaml_data = yaml.dump(json_data, default_flow_style=False)

print("Saving YAML")
#Save YAML
with open("data.yaml", "w") as yaml_file:
    yaml_file.write(yaml_data)

print("JSON data has been converted and saved as YAML.")
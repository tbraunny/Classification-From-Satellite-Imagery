import os
import json

def download_scene(scene_id, download_dir='scenes'):
    # Planet API URL for data download
    SEARCH_URL = 'https://api.planet.com/data/v1/item-types/PSScene4Band/items/'
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Construct the item URL
    item_url = f'{SEARCH_URL}{scene_id}'
    
    # Get item details
    item_response = session.get(item_url)
    if item_response.status_code != 200:
        print(f'Failed to retrieve scene {scene_id}')
        return
    
    item_data = item_response.json()
    
    # Activate the asset (e.g., analytic_sr)
    assets_url = item_data['_links']['assets']
    assets_response = session.get(assets_url)
    if assets_response.status_code != 200:
        print(f'Failed to retrieve assets for scene {scene_id}')
        return
    
    assets = assets_response.json()
    analytic_sr = assets.get('analytic_sr')
    if not analytic_sr:
        print(f'No analytic_sr asset for scene {scene_id}')
        return
    
    # Activate the asset if necessary
    if analytic_sr['status'] != 'active':
        activation_url = analytic_sr['_links']['activate']
        activation_response = session.post(activation_url)
        if activation_response.status_code != 202:
            print(f'Failed to activate asset for scene {scene_id}')
            return
        else:
            print(f'Asset activation initiated for scene {scene_id}')
            # Wait for activation to complete (polling or delay)
    
    # Download the asset
    asset_url = analytic_sr['location']
    asset_response = session.get(asset_url, stream=True)
    if asset_response.status_code != 200:
        print(f'Failed to download asset for scene {scene_id}')
        return
    
    # Save the asset to a file
    scene_path = os.path.join(download_dir, f'{scene_id}.tif')
    with open(scene_path, 'wb') as f:
        for chunk in asset_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f'Scene {scene_id} downloaded successfully.')

# Assuming scene_ids is a list of unique scene IDs from your dataset
unique_scene_ids = set(scene_ids)  # Extract from your dataset
for scene_id in unique_scene_ids:
    download_scene(scene_id)


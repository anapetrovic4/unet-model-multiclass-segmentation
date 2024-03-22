import pandas as pd
import os

# izvuci liste koje ce biti kolone (density, zoom, longitude, latitude, source, rotation)
density = []
zoom = []
longitude = []
latitude = []
source = []
rotation = []

images_path = '/mnt/c/projects/unet-multiclass/krovovi/Standarised_original/Standarised/images/train'

# prodji kroz svaki fajl i kad naidjes _ sve pre toga smesti u listu
def iterate_files():
    for filename in os.listdir(images_path):
        if filename.endswith('.png'):
            
            parts = filename.split('_')
            
            density.append(parts[0].replace('density', ''))
            zoom.append(parts[1].replace('zoom', ''))
            longitude.append(parts[2].replace('longitude', ''))
            latitude.append(parts[3].replace('latitude', ''))
            source.append(parts[4].replace('.png', ''))
            rotation.append(parts[-1].replace('.png', ''))
            
    # ako je u rotation element google, zameni ga sa 0
    
    for i in range(len(rotation)):
        if rotation[i] == 'google' or rotation[i] == 'mapbox':
            rotation[i] = '0'       
    
iterate_files()

print(density[:10])
print(zoom[:10])
print(longitude[:10])
print(source[:10])
print(rotation[:10])


# kreiraj dataframe

data = {
    "density" : density,
    "zoom" : zoom,
    "longitude" : longitude,
    "latitude" : latitude,
    "source" : source,
    "rotation" : rotation
}

df = pd.DataFrame(data)

print(df)

# sacuvaj ga kao csv
df.to_csv('krovovi.csv')
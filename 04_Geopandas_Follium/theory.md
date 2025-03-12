# Análisis de datos geoespaciales con Python

## 1. Introducción al análisis geoespacial de datos 
### 1.1 Geopandas
**GeoPandas** es una biblioteca de Python que extiende las funcionalidades de Pandas para trabajar con datos geoespaciales. Permite manipular, analizar y visualizar datos geográficos de manera eficiente. Los conceptos clave a tener en cuenta cuando usamos esta librería son:

- **Geometry**: los datos geoespaciales se representan como geometrías (**puntos, líneas, polígonos**).
- *DataFrame GeoPandas*: similar a un DataFrame de Pandas, pero con una columna especial llamada geometry que almacena las geometrías.
- **CRS (Coordinate Reference System)**: sistema de referencia espacial que define cómo se proyectan las coordenadas en la Tierra. Se usa el método `.to_crs()` para transformar las coordenadas a un nuevo sistema de referencia. Ejemplo: `gdf = gdf.to_crs(epsg=4326)`

### 1.2 Folium
**Folium** es una biblioteca de Python que permite crear mapas interactivos basados en Leaflet.js. Es ideal para visualizar datos geoespaciales en un mapa web. Los conceptos clave a tener en cuenta cuando usamos esta librería son:

- **Mapa base**: el fondo del mapa (por ejemplo, OpenStreetMap, Stamen Terrain).
`m = folium.Map(location=[40.4165, -3.7026], tiles="Stamen Terrain", zoom_start=6)`
- **Capas**: elementos adicionales que se pueden superponer en el mapa (puntos, polígonos, marcadores, etc.). Se puede usar GeoJson o Choropleth para añadir capas geoespaciales desde un GeoDataFrame de GeoPandas. `gdf_json = json.loads(gdf.to_json())` y después lo pasamos como capa, por ejemplo `folium.Choropleth(geo_data=gdf_json, ...)`
- *Interactividad*: permite hacer zoom, arrastrar el mapa y añadir pop-ups.

Muy recomendado estudiar y analizar **leafmap**, ya que se trata de un mapa interactivo como Folium pero que tiene una grana cantiadd de funcionalidades extra añadidas, lo cual la hacen una opción más completa que folium. En caso de tener datos geoespaciales masivos con muchísima información se recomienda el uso de l alibrería **lonboard** para la visualziación de estos datos.

### 1.3 Principales sistemas de referencia de coordenadas (CRS)
Definen cómo se proyectan las coordenadas en la Tierra y cómo se interpretan los datos geográficos. Un CRS define cómo se mapean las coordenadas geográficas (longitud y latitud) o planas (x, y) en una superficie bidimensional. Hay dos tipos principales de CRS:

- **Geográfico**: basado en longitudes y latitudes, utilizando un elipsoide que representa la forma aproximada de la Tierra. En este caso, el sistema principal es **EPSG:4326** (**WGS84**), el cual usa coordenadas en grados decimales (**latitud y longitud**) y no preserva distancias ni áreas, ya que la Tierra no es plana.
- **Proyectado**: basado en coordenadas planas (x, y), utilizando una proyección cartográfica que transforma la superficie curva de la Tierra en una superficie plana. En este caso el sistema más utilizado es el **UTM** , el cual se define como **EPSG:32630** y divide la Tierra en zonas de 6 grados de longitud, preservando distancias dentro de cada zona. Ideal para análisis espacial que requiere precisión en distancias y áreas. Otro sistema proyectado muy utilizado es el **Web Mercator** que se define con **EPSG:3857**. Este sistema transforma longitudes y latitudes en coordenadas planas (x, y) y es representa el estándar en mapas web como GoogleMaps o OpenStreetMap. Otra de las ventajas que ofrce el CRS de Web Mercator es que las coordenadas planas están fijadas en metros, por lo que aunque distorsiona las distancias reales, ya que estas no se mantienen constantes en toda la proyección, se puede usar para medir distancias si la zona de estudio no es muy grande.

En Python, los CRS suelen definirse mediante **códigos EPSG**. Estos códigos son estándares internacionales que identifican diferentes sistemas de referencia. Para definir el CRS al cargar datos podemos usar `gdf = gdf.to_crs("EPSG:32630")` o `gdf.set_crs(epsg=4326)`. Si por ejemplo hemos cargado los datos como un pandas.DataFrame, podemos usar `gdf = gpd.GeoDataFrame(data=df, geometry = "columna_geometry", crs="EPSG:4326")`.

### 1.4 Shapely
**Shapely** es una biblioteca de Python muy útil para trabajar con geometrías geoespaciales. Está diseñada para manipular y analizar objetos geométricos planos (bidimensionales). Proporciona herramientas para crear, modificar, consultar y analizar geometrías como **puntos, líneas y polígonos**. A diferencia de GeoPandas, que maneja datos tabulares junto con geometrías, Shapely se enfoca exclusivamente en las geometrías en sí mismas. Esto lo hace ideal para tareas específicas de análisis espacial. Usamos shapely para:
- Creación y manipulación de geometrías: `from shapely.geometry import Point, LineString, Polygon`
- Operaciones espaciales avanzadas: `buffer`, `intersection`, `difference`, `union`
- Verificar relaciones entre geometrías: `punto.touches(linea)`, `punto.within(poligono)`, ...
- Medidas y propiedades: `poligono.area`, `linea.length`, `list(punto.coords)`, ....

## 2. GeoPandas
Lo primero y más importante es que, para cargar los datos en un geopandas, usaremos la función `gpd.read_file(path)`, la cual detecta automáticamente el tipo de archivo y crea un GeoDataFrame con los datos que le hemos facilitado. Recordar que **WKT (Well Known Text) se trata de texto que podemos usar para construir geometrías** directamente, usando por ejemplo `geometry = gpd.GeoSeries.fromwkt(data_wkt)`. Comandos básicos importantes:
- `gpd.read_file(path)`: para importar los datos espaciales de un archivo GeoJson, shapefile,... Tener en cuenta que parquet no se encuentra entre la lista de ficheros disponibles, para ello usamos la función especializada `gpd.read_parquet(path)`.
- `GeodataFrame.to_file("name.geojson", driver="GeoJSON")`: para guardar los datos espaciales en un archivo GeoJson, Shapefile, ... Por defecto, se guarda en formato ESRI Shapefile. Para geoparquet o postgis existen otros comandos específicos `GeoDataFrame.to_parquet` y `GeoDataFrame.to_postgis`
- `explore()`: permite iniciar un mapa interactivo en el que se puede visualizar la columna geometry de un geofataframe.
- `gpd.read_postgis(query, con=engine, geom_col="geometry", crs="EPSG:4326")`: para realizar una consultas sobre **PostGIS** y cargar los datos geoespaciales en el notebook. Primero es necesario establecer la conexión a la base de datos.
- `gpd.GeoSeries.fromwkt(data_wkt)`: para transformar una columna WKT en un GeoSeries que pueda actuar como geometry.


### 2.1 Principales funciones geoespaciales
- GeoSeries.area: recordar que la columna geometry debe ser del tipo Polygon. `gdf.geometry.area`
- GeoSeries.length: recordar que la columna geometry debe ser un del tipo LineString. `gdf.geometry.lenght`
- GeoSeries.coords: para extraer las coordenadas x e y. Recordar que geometry debe ser de tipo Point. `list(gdf.geometry.coords)`
- Geoseries.centroid: para extraer las coordenadas del centroide de un Polygobn. `gdf.geometry.centroid`, se obtiene una tupla con centroid.x y centroid.y. Lo que se suele hacer en estos casos es: `centroid = gdf["geometry"].centroid` -> `gdf["longitud"] = centroid.x` + `gdf["latitud"] = centroid.y`.
- Geoseries.boundary: definir los límites de un polygon. Obtenemos una nueva columna con los linestring que representan los límites
- Geoseries.bounds: para ver los coordenadas de los límites de cada uno de los polygon
- Geoseries.total_bounds: para ver las coordenadas de los límites de todos los polygonos
- Geoseries.distance: para calcular la distancia entre dos puntos. Se requiere que las geometrías de ambos sean Point. Por ej: `schools["nearest_subway_distance"] = schools.geometry.apply(lambda school: subways.geometry.distance(school).min())` busca la estación más cercana a cada colegio.
- Geoseries.buffer: se puede usar con cualuquier tipo de forma (Point, Polygon , LineString)
- Geoseries.intersects: para comprobar si existen intersecciones entre formas, ya sea líneas o polígonos. Por ejemplo: `parks["intersects_bike_path"] = parks.geometry.apply(lambda park: bike_paths.geometry.intersects(park).any())`, comprueba si algún carril bici interseca con un parque devolviendo una columna con True o False.
- Geoseries.simplify(tolerance=value): para simplificar nuestra geometría
- Geoseries.overlay(Geoseries, how = ["intersection" or "union" or "identity" or "symetric_difference" or "difference"]): combina las geometrías de ambos GeoDataFrames según el tipo de operación especificado mediante el parámetro how. Itersection se queda con las capas superpuestas, union: combina ambas geometrías, identity: conserva todo de la primera capa y añade lo de la segunda que se superponga,difference: conserva las áreas de la primera capa y las que no se superponen de la segunda y symetric_difference conserva las área sque no se superponen entre las dos capas. por ejemplo `parks_neighborhoods_overlay = neighborhoods.overlay(parks, how="intersection")`
- Geoseries.isvalid: para comprobar si los valores de la geometría son válidos o no.
- Geoseries.x and Geoseries.x: para extraer las corrdenadas x e y (longitud y latitud) de geometrys del tipo Point. `schools["x"] = schools.geometry.to_crs(epsg=4326).x `
- Geoseries.symmetric_difference(Geoseries): conserva las áreas sque no se superponen entre las dos capas
- Geoseries.touches(Geoseries): para comprobar si una  geometría toca a otra, es decir, un objeto toca a otro si tiene al menos un punto en común con el otro y su interior no se cruza con ninguna parte del otro. Tener en cuenta que los elementos superpuestos no se tocan.

### 2.2 Spatial join
Se recomienda definir el **spatial index** `s_index = gdf.sindex`, ya que es mucho más eficiente trabajar con  estos índices cuando queremos hacer un spatial join
- `Geopandas.sjoin(left_df, right_df, how, predicate)`: join esntre dos geodataframes. Por ejmplo, podemos buscar la uniómn entre un punto y un polígono, es decir, buscar los polígono que intersectan con un punto/linestring, por ejmplo. El parámetro `predicate` permite especificar el tipo de relación espacial que debe existir entre las geometrías de ambos dataframes y sus posibles valores son: 'intersects'; join si cualquier parte coincide, 'within'; si una geometría está dentro de la otra y no intersectan entre sí, 'crosses': cuando las geometrías comparten algunos pero no tosos sus puntos, 'touch': si tienen al menos un punto en común pero ningún punto interior y 'overlaps': cuando las geometrías comparten espacio. 
- `Geopandas.sjoin_nearest(left_df, right_df, how, distance_col="distances")`: nos permite obtener una unión espacial basada en la distancia más cercana entre sus puntos. Los resultados incluirán varios registros de salida para un único registro de entrada cuando haya varios vecinos equidistantes más cercanos.


## 3. Folium





# Análisis raster con Python - Rasterio

El **análisis ráster** es un tipo de análisis espacial que se realiza sobre datos en formato ráster (**imágenes satelitales o matrices de celdas**). A diferencia de los datos vectoriales (que representan objetos geográficos como puntos, líneas y polígonos), los datos ráster dividen la superficie terrestre en una cuadrícula de celdas (**píxeles**), donde cada celda tiene un valor asociado (por ejemplo, altitud, temperatura, uso del suelo, etc.). Aunque GeoPandas está diseñado principalmente para trabajar con datos vectoriales, se puede combinar su uso con otras bibliotecas especializadas en análisis ráster, como **Rasterio , Xarray o rioxarray**, para realizar análisis híbridos entre datos ráster y vectoriales.

El análisis ráster implica operaciones y procesamientos sobre datos ráster para extraer información útil. Algunos ejemplos comunes incluyen:
- `Reclasificación`: Cambiar los valores de las celdas según ciertas reglas.
- `Zonal statistics`: Calcular estadísticas (como media, suma, mínimo, máximo) dentro de zonas definidas por datos vectoriales.
- `Superposición`: Combinar múltiples capas ráster para analizar relaciones espaciales.
- `Modelado de visibilidad`: Determinar qué áreas son visibles desde un punto dado.
- `Análisis de pendiente y aspecto`: Calcular características topográficas a partir de un modelo digital de elevación (DEM).

Explorar más sobre las capas de la imágenes atelitales y los tipos de imágenes que hay y las posibles bandas que puede tener cada imagen satelital (por ejemplo las imágenes landsat5.tif tienen 6 capas y las tres primeras se corresponden al espectro visible, es decir, azul, verde y rojo)
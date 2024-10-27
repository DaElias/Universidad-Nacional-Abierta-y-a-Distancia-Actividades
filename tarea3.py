from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, FloatType

# Inicializa la sesion de Spark
spark = SparkSession.builder.appName('Tarea3').getOrCreate()

# Define la ruta del archivo .csv en HDFS
file_path = 'hdfs://localhost:9000/Tarea3/rows.csv'

# Lee el archivo .csv en un DataFrame
df = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(file_path)

# Mostrar el esquema del DataFrame
df.printSchema()

# Realizar una muestra de las primeras filas del DataFrame
print("Vista preliminar de los datos:")
df.show(5)

# Limpieza de datos: eliminar duplicados y filtrar filas nulas en las columnas importantes
df = df.dropDuplicates()
df = df.na.drop(subset=['nombre_entidad', 'municipio', 'numero_de_resolucion', 'valor_sancion'])

# Conversion de tipos: transformar las columnas necesarias a tipos adecuados
df = df.withColumn("valor_sancion", F.col("valor_sancion").cast(FloatType()))
df = df.withColumn("numero_de_contrato", F.col("numero_de_contrato").cast(IntegerType()))

# Analisis exploratorio basico
print("Estadisticas descriptivas de las columnas numericas:")
df.describe(['valor_sancion', 'numero_de_contrato']).show()

# Analisis exploratorio avanzado
# 1. Contar registros por entidad
print("Numero de sanciones por entidad:")
df.groupBy('nombre_entidad').count().orderBy(F.col("count").desc()).show()

# 2. Calcular sancion media por municipio
print("Sancion promedio por municipio:")
df.groupBy('municipio').agg(F.avg('valor_sancion').alias('sancion_promedio')).orderBy(F.col('sancion_promedio').desc()).show()

# 3. Filtrar y seleccionar datos con sanciones superiores a un umbral
print("Filtrar sanciones superiores a 5,000,000:")
dias = df.filter(F.col('valor_sancion') > 5000000).select('nombre_entidad', 'municipio', 'numero_de_resolucion', 'nombre_contratista', 'numero_de_contrato', 'valor_sancion')
dias.show()

# 4. Ordenar datos por el valor de sancion en orden descendente
print("Datos ordenados por valor de sancion de mayor a menor:")
sorted_df = df.sort(F.col("valor_sancion").desc())
sorted_df.show(10)

# Convertir DataFrame a RDD para mostrar su uso (opcional)
rdd = df.rdd

# Mostrar un analisis basico en el RDD
print("Primeros elementos en el RDD:")
print(rdd.take(5))

# Cerrar la sesion de Spark al finalizar
spark.stop()

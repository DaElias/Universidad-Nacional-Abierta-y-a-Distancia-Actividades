import happybase
import pandas as pd
from datetime import datetime  

# Bloque principal de ejecucion
try:
    connection = happybase.Connection('localhost')     
    print("Conexion establecida con HBase")      
    
    table_name = 'productos'     
    families = {
        'product_info': dict(),    # Informacion general del producto
        'product_details': dict(), # Detalles adicionales del producto
        'user_reviews': dict()     # Reseñas de los usuarios
    }
    
    if table_name.encode() in connection.tables():
        print(f"Eliminando tabla existente - {table_name}")
        connection.delete_table(table_name, disable=True)
        
    # Crear nueva tabla
    connection.create_table(table_name, families)
    table = connection.table(table_name)
    print("Tabla 'productos' creada exitosamente")

    # 3. Cargar datos desde el archivo CSV
    product_data = pd.read_csv('amazon.csv')  # Cambia el nombre del archivo si es necesario
    # for index, row in product_data.iterrows():
    cantidad_productos = 150
    for index, row in  product_data.iterrows():
        if index >= cantidad_productos-1:
            break
        row_key = f'product_{index}'.encode()  # Generar row key basado en el índice
        
        # Organizar los datos en las familias de columnas
        data = {
            b'product_info:product_id': str(row['product_id']).encode(),
            b'product_info:product_name': str(row['product_name']).encode(),
            b'product_info:category': str(row['category']).encode(),
            b'product_info:discounted_price': str(row['discounted_price']).encode(),
            b'product_info:actual_price': str(row['actual_price']).encode(),
            b'product_info:discount_percentage': str(row['discount_percentage']).encode(),
            b'product_info:rating': str(row['rating']).encode(),
            b'product_info:rating_count': str(row['rating_count']).encode(),
            
            b'product_details:about_product': str(row['about_product']).encode(),
            b'product_details:img_link': str(row['img_link']).encode(),
            b'product_details:product_link': str(row['product_link']).encode(),
            # Aquí suponemos que las reseñas están en las columnas relacionadas
            b'user_reviews:user_id': str(row['user_id']).encode(),
            b'user_reviews:user_name': str(row['user_name']).encode(),
            b'user_reviews:review_id': str(row['review_id']).encode(),
            b'user_reviews:review_title': str(row['review_title']).encode(),
            b'user_reviews:review_content': str(row['review_content']).encode()
        }
        
        # Insertar los datos en HBase
        table.put(row_key, data)
        
    print("Datos cargados exitosamente")

    # 4. Consultas y análisis de datos
    print("\n=== Todos los productos en la base de datos (primeros 3) ===")
    count = 0
    for key, data in table.scan():
        if count < 3:  # Limitamos a 3 para el ejemplo
            # print(f"\nProducto ID: {key.decode()}")
            print(f"Nombre: {data[b'product_info:product_name'].decode()}")
            print(f"Categoría: {data[b'product_info:category'].decode()}")
            print(f"Precio con descuento: {data[b'product_info:discounted_price'].decode()}\n")
            count += 1
    
    # 5. Encontrar productos con descuento mayor al 30%
    print("\n=== Productos con descuento mayor al 30% ===")
    count = 0
    for key, data in table.scan():
        if float(data[b'product_info:discount_percentage'].decode().replace("%","")) > 30:
            count += 1
            print(f"Nombre: {data[b'product_info:product_name'].decode()}")
            # print(f"Descuento: {data[b'product_info:discount_percentage'].decode()}%")
    print("\n=== Cantidad de Productos con descuento mayor al 30%: ",count)
    print("\n=== Cantidad de Productos que no aplican descuento  : ",cantidad_productos-count)
    
    # 6. Análisis de calificacion por categoría
    print("\n=== Procentaje de productos cons descuentos mayores al 30% ===")
    print("\n=== procentaje:  ", (count/cantidad_productos)*100, "% \n")

    print("\n=== Promedio de calificacion por categoría ===")
    category_ratings = {}
    category_counts = {}
    for key, data in table.scan():
        try:
            category = data[b'product_info:category'].decode()
            rating = float(data[b'product_info:rating'].decode())
            category_ratings[category] = category_ratings.get(category, 0) + rating
            category_counts[category] = category_counts.get(category, 0) + 1
        except ValueError:
            print(f"\nValueError: ",ValueError)
        
    for category in category_ratings:
        avg_rating = category_ratings[category] / category_counts[category]
        print(f"{category}: {avg_rating:.2f}")
    
    # 7. Ejemplo de actualizacion de precio
    product_to_update = 'product_0'
    new_price = 75000
    table.put(product_to_update.encode(), {b'product_info:discounted_price': str(new_price).encode()}) 
    print(f"\nPrecio actualizado para el producto ID: {product_to_update}")

except Exception as e:
    print(f"Error: {str(e)}") 
finally: # Cerrar la conexion 
    connection.close() 

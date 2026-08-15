# Estimación del precio de una vivienda
## Introducción
En este proyecto, nos propusimos dar respuesta a la siguiente pregunta
¿Cómo es posible estimar el precio de una vivienda?
Para ello, nos enfocamos un caso puntual que nos permitiera estudiar la
relación entre distintas características de la vivienda y desarrollar un
modelo de regresión pequeño que sea fácil de implementar y con buena
interpretabilidad.

### El problema de estimar el precio
¿Cómo es posible estimar el precio de una vivienda?
Nuestro objetivo fue desarrollar un programa que nos ayudase a responder
esta pregunta y a realizar las estimaciones. Nuestro enfoque consiste en
definir una variable objetivo, el precio de la vivienda, y estudiar esta
variable en función de otras variables asociadas a la vivienda, como pueden
ser características geográficas, demográficas, o bien características
propias de la vivienda.

Si bien el problema es un simple problema de regresión, dada la complejidad
de las relaciones entre las variables, tomamos la decisión de abordar el
problema mediante arquitecturas de Deep Learning. Por ejemplo,
la característica más fuertemente correlacionada con el precio de la vivienda
es el ingreso promedio. Sin embargo, un vistazo al mapa de California y al
gráfico del precio de la vivienda como función de la ubicación
(latitud, longitud) parece indicar que la presencia de centros urbanos es
un factor importante también, aunque esta dependencia no quede clara haciendo
un análisis de correlación.

Teniendo esto en cuenta, hemos probado modelos de distinto grado de
complejidad, en busca de un modelo que sea a la vez potente en cuanto a
sus predicciones, simple de explicar y de bajo costo de implementación y
mantenimiento.

### Desarrollo del producto final
Una vez obtenido un modelo satisfactorio, desarrollamos una interfaz que
permite, con información parcial, dar una estimación del precio de una
vivienda. Dicha interfaz está basada en el conjunto de datos utilizados
para desarrollar los modelos mencionados anteriormente, pero creemos que
la metodología utilizada en este caso puntual es adaptable a otros conjuntos
de datos.

En una situación de uso real, los modelos deberán entrenarse con datos
adecuados. A su vez, dado que los mercados inmobiliarios son cambiantes,
estos modelos requieren ser reentrenados de manera periódica para obtener
buenos resultados.

### Preguntas relacionadas y productos derivados
A lo largo del desarrollo del proyecto, implementamos distintas arquitecturas,
desde modelos de regresión simples, pasando por modelos de perceptrones
multicapa y redes neuronales densas, hasta un modelo de atención secuencial
(TabNet).

En pocas palabras, los modelos MLP y DNN implementados parecen dar mejores
resultados que el modelo TabNet que entrenamos. Atribuimos esto a la baja
dimensionalidad del dataset utilizado y la disponibilidad de datos
etiquetados; la componente central de TabNet, su mecanismo de atención, no
ofrece suficientes ventajas, comparadas con su costo de implementación,
a la hora de realizar predicciones del precio de la vivienda.

Sin embargo, el enfoque proporcionado por el modelo TabNet sugiere otro
problema y un posible producto derivado del mismo: recomendar a un usuario
una ubicación en función del target de precio de la vivienda y características
demográficas deseadas del entorno, o bien características de la vivienda
misma.

## Metodología
### Datos utilizados
El conjunto de datos utilizado para estudiar el problema y entrenar los
modelos propuestos es [California Housing Dataset](https://www.google.com/url?q=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fdatasets%2Freal_world.html%23california-housing-dataset), un conjunto de datos con información acerca de viviendas en el estado de California, Estados Unidos.
Este conjunto de datos consiste en una tabla de 20.640 entradas, organizada en
nueve columnas:
- `MedHouseVal`: la variable objetivo, el valor medio de la vivienda por bloque censal en California, expresado en cientos de miles de dólares.
- `MedInc`: ingreso medio de los habitantes del bloque;
- `HouseAge`: antigüedad media de las viviendas del bloque;
- `AveRooms`: promedio de habitaciones por hogar;
- `AveBedrms`: promedio de dormitorios por hogar;
- `Population`: población total del bloque;
- `AveOccup`: promedio de ocupantes por hogar;
- `Latitude`: latitud del bloque;
- `Longitude`: longitud del bloque.

### Fases de desarrollo del proyecto
En una primera instancia, realizamos un análisis exploratorio del
conjunto de datos. Entre los resultados de este análisis inicial,
destacamos que el ingreso medio por bloque censal (`MedInc`) presenta
una asociación lineal más fuerte con el precio de la vivienda
(R=0,688), mientras que el resto de los predictores exhibe una correlación
lineal débil (R<0,2). Esto no implica que estas variables no tengan poder
predictivo, sino que, su aporte puede presentarse de manera no lineal o
en combinación con otras variables.

Para capturar estas posibles interacciones, optamos por arquitecturas de
DL, comenzando por la implementación de un MLP. En esta etapa, realizamos
una preparación (normalización de las características) y división de los
datos (separación en conjuntos de train y test). Esta estructura fue
replicada para entrenar otros modelos, por lo que la implementamos en un
módulo separado. El *mean squared error* (MSE) estimado de nuestro mejor
MLP fue de 0,269.

En busca de mejores resultados, implementamos una DNN. Si bien
los resultados sólo mejoraron levemente, obteniendo un MSE de 0,257,
las herramientas utilizadas en este caso ofrecen mayor flexibilidad.
La diferencia con el modelo MLP considerado previamente es, esencialmente,
la profundidad de la red. Aunque esto pueda conllevar un mayor costo de
entrenamiento, consideramos que los mejores resultados hacen que valga
la pena.

### Estructura del código
- [Conjunto de datos](data),
- [Estadísticas y gráficas del conjunto de datos](estadisticas),
- [Preparación de los datos para entrenamiento](scripts/preparacion_de_datos.py),
- [Implementación de la arquitectura](scripts/implementacion_de_la_arquitectura.py),
- [Optimización de los hiperparámetros](scripts/optimizacion_de_los_hiperparametros.py),
- [Entrenamiento del modelo](scripts/entrenamiento_del_modelo.py),
- [Métricas y gráficas de rendimiento del modelo](metricas),

### Herramientas utilizadas
Para la implementación de la arquitectura detrás del modelo de DL utilizado
en el producto final, utilizamos distintas componentes de la librería
`TensorFlow`. Para el desarrollo de la interfaz y la presentación del
producto, utilizamos `gradio`.

## Diseño del modelo
Se seleccionó una DNN para este proyecto final, continuando con el dataset California Housing, ya que ofrece un buen equilibrio entre capacidad predictiva y costo computacional frente a las otras arquitecturas evaluadas: en las prácticas anteriores superó tanto al MLP como a TabNet en las cuatro métricas evaluadas (MSE 0,2569, RMSE 0,5068, MAE 0,3480, R² 0,8040), con un tiempo de entrenamiento considerablemente menor al de TabNet.

## Conclusiones


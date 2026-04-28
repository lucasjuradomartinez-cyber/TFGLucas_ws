Las 3 terminales que hay que tener abiertas son: 

Terminal 1: El Driver de la ZED 2 Aquí despertamos el hardware de la cámara para que empiece a emitir en /zed/zed_node/rgb/color/rect/image.
source ~/TFGLucas_ws/install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2


Terminal 2: Tu IA (El Nuevo Nodo ZED) Aquí lanzas el nodo nuevo que creamos, que ya sabe perfectamente dónde ir a buscar la imagen de la ZED sin necesidad de remapeos raros.
source ~/TFGLucas_ws/install/setup.bash
ros2 run sarnet_py zed_segmentation_node_v2


Terminal 3: El Visualizador
source ~/TFGLucas_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view

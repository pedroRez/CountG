from roboflow import Roboflow

# Substitua 'YOUR_API_KEY' pela sua chave de API
rf = Roboflow(api_key="Frxb5HokQQmSHSjhneRy")

# Acesse o projeto e versão específicos
project = rf.workspace("ali-khalili").project("cattle-body-parts-dataset-for-object-detection")
dataset = project.version(4).download("yolov8")
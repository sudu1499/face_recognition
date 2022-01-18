import yaml

from save_detected_face import save_detected__faces
import save_detected_face
config=yaml.safe_load(open('config.yaml','r'))

save_detected__faces(config)
save_detected_face.save_detected__faces(config)
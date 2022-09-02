# path = "models/FSRCNN_x2.pb"
# sr.readModel(path)
# superres_zoomrate=2
# sr.setModel("fsrcnn", superres_zoomrate) # set the model by passing the value and the upsampling ratio

# path = "export_lapsrn/LapSRN_x2.pb"
# sr.readModel(path)
# superres_zoomrate=2
# sr.setModel("lapsrn", superres_zoomrate) # set the model by passing the value and the upsampling ratio

# path = "models_edsr/EDSR_x2.pb"
# sr.readModel(path)
# superres_zoomrate=2
# sr.setModel("edsr", superres_zoomrate) # set the model by passing the value and the upsampling ratio

# For example ESPCN:

import cv2

#frame=..
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "export/ESPCN_x2.pb"
sr.readModel(path)
superres_zoomrate=2
sr.setModel("espcn", superres_zoomrate) # set the model by passing the value and the upsampling ratio

result = sr.upsample(frame) # upscale the input image

cv2.imshow('Zoomed Frame',result)


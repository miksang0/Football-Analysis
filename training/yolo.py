from ultralytics import YOLO
model = YOLO('yolo26m.pt')
drive_path = '/content/drive/MyDrive/Football Analysis/output_results'

results = model.predict('inputvideos/football.mp4', save = True)
print(results[0])
print('******************************')
for box in results[0].boxes:
  print(box)
  
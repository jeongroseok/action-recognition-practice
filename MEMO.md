# 1차 시도
test centernet
extract centerpoint only
extract padded sq box
ctdet=centernet detection

# 2차 시도
detect via yolo
group bbox(겹치면 하나로, 안겹치면 분리)
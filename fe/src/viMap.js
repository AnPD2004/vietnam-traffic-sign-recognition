// Mapping from class_id (number) and English class name to Vietnamese label.
// Generated from data/vn-traffic-signs/classes_en.txt and classes_vie.txt
const VI_MAP = {
  0: 'Đường người đi bộ cắt ngang',
  'Pedestrian Crossing': 'Đường người đi bộ cắt ngang',

  1: 'Đường giao nhau (ngã ba bên phải)',
  'Equal-level Intersection': 'Đường giao nhau (ngã ba bên phải)',

  2: 'Cấm đi ngược chiều',
  'No Entry': 'Cấm đi ngược chiều',

  3: 'Phải đi vòng sang bên phải',
  'Right Turn Only': 'Phải đi vòng sang bên phải',

  4: 'Giao nhau với đường đồng cấp',
  'Intersection': 'Giao nhau với đường đồng cấp',

  5: 'Giao nhau với đường không ưu tiên',
  'Intersection with a non-priority road': 'Giao nhau với đường không ưu tiên',

  6: 'Chỗ ngoặt nguy hiểm vòng bên trái',
  'Danger zone on the left': 'Chỗ ngoặt nguy hiểm vòng bên trái',

  7: 'Cấm rẽ trái',
  'No Left Turn': 'Cấm rẽ trái',

  8: 'Bến xe buýt',
  'Bus Stop': 'Bến xe buýt',

  9: 'Nơi giao nhau chạy theo vòng xuyến',
  'Roundabout': 'Nơi giao nhau chạy theo vòng xuyến',

  10: 'Cấm dừng và đỗ xe',
  'No Stopping and No Parking': 'Cấm dừng và đỗ xe',

  11: 'Chỗ quay xe',
  'U-Turn Allowed': 'Chỗ quay xe',

  12: 'Biển gộp làn đường theo phương tiện',
  'Lane Allocation': 'Biển gộp làn đường theo phương tiện',

  13: 'Đi chậm',
  'Slow Down': 'Đi chậm',

  14: 'Cấm xe tải',
  'No Trucks Allowed': 'Cấm xe tải',

  15: 'Đường bị thu hẹp về phía phải',
  'Narrow Road on the Right': 'Đường bị thu hẹp về phía phải',

  16: 'Giới hạn chiều cao',
  'Height Limit': 'Giới hạn chiều cao',

  17: 'Cấm quay đầu',
  'No U-Turn': 'Cấm quay đầu',

  18: 'Cấm ô tô khách và ô tô tải',
  'No Passenger Cars and Trucks': 'Cấm ô tô khách và ô tô tải',

  19: 'Cấm rẽ phải và quay đầu',
  'No U-Turn and No Right Turn': 'Cấm rẽ phải và quay đầu',

  20: 'Cấm ô tô',
  'No Cars Allowed': 'Cấm ô tô',

  21: 'Đường bị thu hẹp về phía trái',
  'Narrow Road on the Left': 'Đường bị thu hẹp về phía trái',

  22: 'Gồ giảm tốc phía trước',
  'Uneven Road': 'Gồ giảm tốc phía trước',

  23: 'Cấm xe hai và ba bánh',
  'No Two or Three-wheeled Vehicles': 'Cấm xe hai và ba bánh',

  24: 'Kiểm tra',
  'Customs Checkpoint': 'Kiểm tra',

  25: 'Chỉ dành cho xe máy*',
  'Motorcycles Only': 'Chỉ dành cho xe máy*',

  26: 'Chướng ngoại vật phía trước',
  'Obstacle on the Road': 'Chướng ngoại vật phía trước',

  27: 'Trẻ em',
  'Children Present': 'Trẻ em',

  28: 'Xe tải và xe công*',
  'Trucks and Containers': 'Xe tải và xe công*',

  29: 'Cấm mô tô và xe máy',
  'No Motorcycles Allowed': 'Cấm mô tô và xe máy',

  30: 'Chỉ dành cho xe tải*',
  'Trucks Only': 'Chỉ dành cho xe tải*',

  31: 'Đường có camera giám sát',
  'Road with Surveillance Camera': 'Đường có camera giám sát',

  32: 'Cấm rẽ phải',
  'No Right Turn': 'Cấm rẽ phải',

  33: 'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải',
  'Double curve first to right': 'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải',

  34: 'Cấm xe sơ-mi rơ-moóc',
  'No Containers Allowed': 'Cấm xe sơ-mi rơ-moóc',

  35: 'Cấm rẽ trái và phải',
  'No Left or Right Turn': 'Cấm rẽ trái và phải',

  36: 'Cấm đi thẳng và rẽ phải',
  'No Straight and Right Turn': 'Cấm đi thẳng và rẽ phải',

  37: 'Đường giao nhau (ngã ba bên trái)',
  'Intersection with T-Junction': 'Đường giao nhau (ngã ba bên trái)',

  38: 'Giới hạn tốc độ (50km/h)',
  'Speed limit (50km/h)': 'Giới hạn tốc độ (50km/h)',

  39: 'Giới hạn tốc độ (60km/h)',
  'Speed limit (60km/h)': 'Giới hạn tốc độ (60km/h)',

  40: 'Giới hạn tốc độ (80km/h)',
  'Speed limit (80km/h)': 'Giới hạn tốc độ (80km/h)',

  41: 'Giới hạn tốc độ (40km/h)',
  'Speed limit (40km/h)': 'Giới hạn tốc độ (40km/h)',

  42: 'Các xe chỉ được rẽ trái',
  'Left Turn': 'Các xe chỉ được rẽ trái',

  43: 'Chiều cao tĩnh không thực tế',
  'Low Clearance': 'Chiều cao tĩnh không thực tế',

  44: 'Nguy hiểm khác',
  'Other Danger': 'Nguy hiểm khác',

  45: 'Đường một chiều',
  'One-way street': 'Đường một chiều',

  46: 'Cấm đỗ xe',
  'No Parking': 'Cấm đỗ xe',

  47: 'Cấm ô tô quay đầu xe (được rẽ trái)',
  'No U-Turn for Cars': 'Cấm ô tô quay đầu xe (được rẽ trái)',

  48: 'Giao nhau với đường sắt có rào chắn',
  'Level Crossing with Barriers': 'Giao nhau với đường sắt có rào chắn',

  49: 'Cấm rẽ trái và quay đầu xe',
  'No U-Turn and No Left Turn': 'Cấm rẽ trái và quay đầu xe',

  50: 'Chỗ ngoặt nguy hiểm vòng bên phải',
  'Danger zone on the right': 'Chỗ ngoặt nguy hiểm vòng bên phải',

  51: 'Chú ý chướng ngại vật – vòng tránh sang bên phải',
  'Warning: Obstacle ahead – pass on the right': 'Chú ý chướng ngại vật – vòng tránh sang bên phải',
}

export default VI_MAP

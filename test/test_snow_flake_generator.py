# 创建一个生成器，指定数据中心ID和工作机器ID
from utils.snow_flake_generator import generate_unique_id

datacenter_id = 1
worker_id = 1

# 生成用户ID
user_id = generate_unique_id("USER_", datacenter_id, worker_id)
print(f"生成的用户ID: {user_id}")

# 生成订单ID
order_id = generate_unique_id("ORDER_", datacenter_id, worker_id)
print(f"生成的订单ID: {order_id}")

# 生成多个产品ID
for i in range(5):
    product_id = generate_unique_id("PROD_", datacenter_id, worker_id)
    print(f"产品 {i + 1} ID: {product_id}")
